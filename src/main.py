import random

import hydra
import pandas
import torch
from omegaconf import DictConfig, OmegaConf

from src.data import dataset_dict, get_dataloader, mean_dict, std_dict
from src.evaluation import (
    classification_eval,
    segmentation_eval,
    depth_estimation_eval,
)
from src.models import (
    LinearClassifier,
    SegmenterHead,
    Segmenter,
    BNHead,
    DepthEncoderDecoder,
    load_ssl_models,
    encoder_sanity_check,
)
from src.train import train_classifier, train_segmenter, train_depth_estimator
from src.utils import get_writer


@hydra.main(version_base=None, config_path="../conf/")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    action_cfg = cfg.action
    task_cfg = cfg.task
    ssl_model_cfg = cfg.ssl_model

    # set seed for reproducibility
    torch.manual_seed(action_cfg.seed)
    torch.cuda.manual_seed(action_cfg.seed)
    random.seed(action_cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(action_cfg.device)
    class_cnt = dataset_dict[task_cfg.dataset].classes_cnt
    dataset_name = dataset_dict[task_cfg.dataset].name

    print("Start", ssl_model_cfg.name, task_cfg.dataset)
    writer = get_writer(
        {
            "action": OmegaConf.to_container(action_cfg),
            "task": OmegaConf.to_container(task_cfg),
            "ssl_model": OmegaConf.to_container(ssl_model_cfg),
        }
    )
    if action_cfg.action == "downstream_training":
        print("downstream_training")
        result = train_downstream(
            action_cfg, task_cfg, ssl_model_cfg, device, class_cnt, dataset_name
        )
        writer.write_pandas(f"acc", pandas.DataFrame(result))
    if action_cfg.action == "adversarial_attack":
        print("adversarial_attack")
        adv_acc = evaluate_adversarially(
            action_cfg, task_cfg, ssl_model_cfg, device, class_cnt
        )
        writer.write_pandas(f"adv_acc", pandas.DataFrame(adv_acc))


def train_downstream(
    action_cfg: DictConfig,
    task_cfg: DictConfig,
    ssl_model_cfg: DictConfig,
    device: torch.device,
    class_cnt: int,
    dataset_name: str,
):
    train_loader = get_dataloader(
        task_cfg.dataset,
        action_cfg.data_root,
        batch_size=task_cfg.batch_size,
        two_views=False,
        augment=False,
        split="train",
        crop_size=ssl_model_cfg.image_size,
        num_workers=action_cfg.num_workers,
        normalize=ssl_model_cfg.normalize if "normalize" in ssl_model_cfg else True,
    )
    eval_loader = get_dataloader(
        task_cfg.dataset,
        action_cfg.data_root,
        batch_size=(
            1 if task_cfg.task != "classification" else task_cfg.batch_size
        ),  # because of sliding window inference
        two_views=False,
        augment=False,
        split="test",
        crop_size=ssl_model_cfg.image_size,
        num_workers=action_cfg.num_workers,
        normalize=ssl_model_cfg.normalize if "normalize" in ssl_model_cfg else True,
    )

    path = ssl_model_cfg.ckpt.final_ckpt
    print(f"Loading SSL Encoder from path: {path}")
    encoder = load_ssl_models(
        ssl_model_cfg,
        load_weights=True,
        ckpt_path=path,
    ).encoder.to(device)
    if task_cfg.task == "classification":
        clf = LinearClassifier(
            encoder,
            task_cfg.dataset,
            task_cfg.lr,
            task_cfg.epochs,
            class_cnt,
            device=device,
            is_train=True,
        )
        val_acc = train_classifier(clf, train_loader, eval_loader)
        encoder_sanity_check(ssl_model_cfg, path, device, clf)
        print(f"Classification {val_acc=}")

    elif task_cfg.task == "segmentation":
        head = SegmenterHead(
            class_cnt,
            encoder,
            dataset_name=dataset_name,
            epochs=task_cfg.epochs,
            lr=task_cfg.lr,
        ).to(device)
        accuracy, miou = train_segmenter(
            task_cfg=task_cfg,
            model_cfg=ssl_model_cfg,
            head=head,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            encoder=encoder,
            device=device,
            n_labels=class_cnt,
        )
        encoder_sanity_check(
            ssl_model_cfg,
            path,
            device,
            Segmenter(num_labels=class_cnt, encoder=encoder, head=head),
        )
        print(f"Segmentation {accuracy=}, {miou=}")

    elif task_cfg.task == "depth_estimation":
        decode_head = BNHead(
            encoder, task_cfg, ssl_model_cfg, ckpt_path=path, device=device
        )
        encoder_decoder = DepthEncoderDecoder(encoder, decode_head)
        rmse = train_depth_estimator(
            encoder_decoder,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            task_cfg=task_cfg,
            device=device,
        )
        encoder_sanity_check(ssl_model_cfg, path, device, encoder_decoder)
        print(f"Depth estimation {rmse=}")
    else:
        raise ValueError("Task not supported")


def evaluate_adversarially(
    action_cfg: DictConfig,
    task_cfg: DictConfig,
    ssl_model_cfg: DictConfig,
    device: torch.device,
    class_cnt: int,
) -> tuple[float, float]:
    normalize = ssl_model_cfg.normalize if "normalize" in ssl_model_cfg else True
    mean = mean_dict["imagenet" if normalize else "no_norm"]
    std = std_dict["imagenet" if normalize else "no_norm"]

    eval_loader = get_dataloader(
        task_cfg.dataset,
        action_cfg.data_root,
        batch_size=task_cfg.batch_size if task_cfg.task == "classification" else 1,
        two_views=False,
        augment=False,
        split="test",
        crop_size=ssl_model_cfg.image_size,
        normalize=normalize,
    )

    path = ssl_model_cfg.ckpt.final_ckpt
    print(f"Loading SSL Encoder from path: {path}")
    encoder = load_ssl_models(
        ssl_model_cfg,
        load_weights=True,
        ckpt_path=path,
    ).encoder.to(device)

    args = [
        action_cfg,
        task_cfg,
        device,
        encoder,
        eval_loader,
        normalize,
        mean,
        std,
    ]

    if task_cfg.task == "classification":
        args.append(class_cnt)
        return classification_eval(*args)

    elif task_cfg.task == "segmentation":
        args.extend([ssl_model_cfg, class_cnt])
        return segmentation_eval(*args)

    elif task_cfg.task == "depth_estimation":
        args.extend([ssl_model_cfg, path])
        return depth_estimation_eval(*args)

    raise Exception(
        f'Wrong task name {task_cfg.task=}, allowed: ["classification", "segmentation", "depth_estimation"]'
    )


if __name__ == "__main__":
    main()

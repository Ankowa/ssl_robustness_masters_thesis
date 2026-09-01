import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.attack import (
    SegPGD,
    EmbedAttack,
    ClassificationPGD,
    DepthPGD,
)

from src.models import (
    LinearClassifier,
    Segmenter,
    SegmenterHead,
    Encoder,
    DepthEncoderDecoder,
    BNHead,
)

from src.evaluation.adversarial import (
    adversarial_classification_accuracy,
    adversarial_depth_estimation,
    adversarial_segmenter_accuracy_miou,
)


def classification_eval(
    action_cfg: DictConfig,
    task_cfg: DictConfig,
    device: torch.device,
    encoder: Encoder,
    eval_loader: DataLoader,
    mean: tuple[int, int, int],
    std: tuple[int, int, int],
    class_cnt: int,
) -> tuple[float, float]:
    clf = LinearClassifier(
        encoder,
        task_cfg.dataset,
        task_cfg.lr,
        task_cfg.epochs,
        class_cnt,
        device=device,
        is_train=True,
    )
    if clf.exists_clf():
        clf.load_clf()
    else:
        raise RuntimeError("Please train the classifier before the attack")
    clf.encoder.unfreeze()

    # EmbedAttack
    encoder_attack = EmbedAttack(
        loss=torch.nn.MSELoss(),
        num_steps=action_cfg.attack_num_steps,
        eps_budget=action_cfg.attack_epsilon,
        lr=action_cfg.attack_lr,
        mean=mean,
        std=std,
    )
    encoder_adv_acc = adversarial_classification_accuracy(
        encoder_attack, encoder, eval_loader, device, clf
    )

    # PGD
    downstream_attack = ClassificationPGD(
        num_steps=action_cfg.attack_num_steps,
        eps_budget=action_cfg.attack_epsilon,
        lr=action_cfg.attack_lr,
        mean=mean,
        std=std,
    )
    downstream_adv_acc = adversarial_classification_accuracy(
        downstream_attack,
        encoder,
        eval_loader,
        device,
        clf,
        task_specific=True,
    )
    print(
        f"encoder_adv_acc: {encoder_adv_acc}, "
        f"downstream_adv_acc: {downstream_adv_acc}, ",
        end="",
    )
    return encoder_adv_acc, downstream_adv_acc


def segmentation_eval(
    action_cfg: DictConfig,
    task_cfg: DictConfig,
    device: torch.device,
    encoder: Encoder,
    eval_loader: DataLoader,
    mean: tuple[int, int, int],
    std: tuple[int, int, int],
    ssl_model_cfg: DictConfig,
    class_cnt: int,
) -> tuple[float, float]:
    head: SegmenterHead = SegmenterHead(
        class_cnt,
        encoder,
        dataset_name=task_cfg.dataset,
        epochs=task_cfg.epochs,
        lr=task_cfg.lr,
    ).to(device)
    if head.exists_segmenter():
        head.load_segmenter()
    else:
        raise RuntimeError("Please train the segmenter before the attack")
    segmenter = Segmenter(class_cnt, encoder, head).to(device)
    encoder.unfreeze()
    encoder.eval()
    head.eval()

    # EmbedAttack
    encoder_attack = EmbedAttack(
        loss=torch.nn.MSELoss(),
        num_steps=action_cfg.attack_num_steps,
        eps_budget=action_cfg.attack_epsilon,
        lr=action_cfg.attack_lr,
        mean=mean,
        std=std,
    )
    _, encoder_adv_miu = adversarial_segmenter_accuracy_miou(
        encoder_attack,
        encoder,
        eval_loader,
        device,
        segmenter,
        n_labels=class_cnt,
        model_cfg=ssl_model_cfg,
        task_cfg=task_cfg,
    )

    # SegPGD
    downstream_attack = SegPGD(
        loss=torch.nn.CrossEntropyLoss(ignore_index=0),  # 0 is for background
        num_steps=action_cfg.attack_num_steps,
        eps_budget=action_cfg.attack_epsilon,
        lr=action_cfg.attack_lr,
        mean=mean,
        std=std,
    )
    (
        _,
        downstream_adv_miu,
    ) = adversarial_segmenter_accuracy_miou(
        downstream_attack,
        encoder,
        eval_loader,
        device,
        segmenter,
        n_labels=class_cnt,
        downstream=True,
        model_cfg=ssl_model_cfg,
        task_cfg=task_cfg,
    )
    print(
        f"encoder_adv_miou: {encoder_adv_miu}, downstream_adv_miu: {downstream_adv_miu}",
        end="",
    )
    return encoder_adv_miu, downstream_adv_miu


def depth_estimation_eval(
    action_cfg: DictConfig,
    task_cfg: DictConfig,
    device: torch.device,
    encoder: Encoder,
    eval_loader: DataLoader,
    mean: tuple[int, int, int],
    std: tuple[int, int, int],
    ssl_model_cfg: DictConfig,
    path: str,
) -> tuple[float, float]:
    decode_head = BNHead(
        encoder, task_cfg, ssl_model_cfg, ckpt_path=path, device=device
    )
    encoder_decoder = DepthEncoderDecoder(encoder, decode_head)
    if decode_head.exists():
        decode_head.load()
    else:
        raise RuntimeError("Please train the depther before the attack")
    encoder.unfreeze()
    encoder.eval()
    decode_head.conv_depth.eval()

    # EmbedAttack
    encoder_attack = EmbedAttack(
        loss=torch.nn.MSELoss(),
        num_steps=action_cfg.attack_num_steps,
        eps_budget=action_cfg.attack_epsilon,
        lr=action_cfg.attack_lr,
        mean=mean,
        std=std,
    )
    encoder_adv_rmse = adversarial_depth_estimation(
        attack=encoder_attack,
        data_loader=eval_loader,
        device=device,
        encoder_decoder=encoder_decoder,
        task_cfg=task_cfg,
        downstream=False,
    )

    # DepthPGD
    downstream_attack = DepthPGD(
        loss=None,
        num_steps=action_cfg.attack_num_steps,
        eps_budget=action_cfg.attack_epsilon,
        lr=action_cfg.attack_lr,
        mean=mean,
        std=std,
    )
    downstream_adv_rmse = adversarial_depth_estimation(
        attack=downstream_attack,
        data_loader=eval_loader,
        device=device,
        encoder_decoder=encoder_decoder,
        task_cfg=task_cfg,
        downstream=True,
    )
    print(
        f"encoder_adv_rmse: {encoder_adv_rmse}, "
        f"downstream_adv_rmse: {downstream_adv_rmse}, ",
        end="",
    )
    return encoder_adv_rmse, downstream_adv_rmse

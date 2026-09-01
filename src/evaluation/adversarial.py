import torch

from src.attack import Attack
from src.evaluation.utils import (
    get_macc_miou,
    sliding_window_inference,
    get_patch_embeddings_pgd,
)
from src.models import Encoder, Segmenter, LinearClassifier, DepthEncoderDecoder

from tqdm import tqdm
from torch.utils.data import DataLoader


def adversarial_classification_accuracy(
    attack: Attack,
    encoder,
    data_loader: DataLoader,
    device,
    classifier: LinearClassifier,
    task_specific: bool = False,
):
    correct = 0
    total = 0
    for idx, (images, labels) in tqdm(enumerate(data_loader)):
        if task_specific:  # PGD
            _, success = attack.run(
                images=images.to(device),
                labels=labels.to(device),
                model=classifier,
            )
            correct += labels.size(0) - success.sum().item()
        else:  # EmbedAttack
            adv_img, _ = attack.run(
                images=images.to(device),
                model=encoder,
            )
            outputs = classifier(adv_img)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.to(device)).sum().item()
        total += labels.size(0)
        if (idx + 1) * data_loader.batch_size >= 1000:
            break
    return correct / total


def adversarial_segmenter_accuracy_miou(
    attack: Attack,
    encoder: Encoder,
    data_loader: DataLoader,
    device,
    segmenter: Segmenter,
    n_labels: int,
    model_cfg: dict,
    task_cfg: dict,
    task_specific: bool = False,
) -> float:
    get_patch_embeddings = get_patch_embeddings_pgd(
        segmenter=segmenter,
        encoder=encoder,
        n_labels=n_labels,
        device=device,
        model_cfg=model_cfg,
        task_cfg=task_cfg,
    )

    predictions, labels_all = [], []
    for _, (images, labels) in tqdm(enumerate(data_loader)):
        if task_specific:  # SegPGD
            adv_img = attack.run(
                image=images.to(device),
                model=segmenter,
                label=labels,
                device=device,
                model_cfg=model_cfg,
                task_cfg=task_cfg,
            )
        else:  # EmbedAttack
            adv_img, _ = attack.run(
                images=images.to(device),
                model=encoder,
                get_representation=get_patch_embeddings,
            )
        with torch.no_grad():
            preds = (
                sliding_window_inference(
                    model=segmenter,
                    num_classes=n_labels,
                    input_data=adv_img.to(device),
                    crop_size=(
                        segmenter.encoder.image_size,
                        segmenter.encoder.image_size,
                    ),
                    stride=(model_cfg.stride_size, model_cfg.stride_size),
                    batch_size=task_cfg.batch_size,
                    return_logits=True,
                )
                .cpu()
                .argmax(dim=1)
            )

        predictions += [pred for pred in preds]
        labels_all += [label for label in labels]

    metrics = get_macc_miou(predictions, labels_all, n_labels)
    return metrics["mean_accuracy"], metrics["mean_iou"]


def adversarial_depth_estimation(
    attack: Attack,
    data_loader: DataLoader,
    device,
    encoder_decoder: DepthEncoderDecoder,
    task_cfg,
    task_specific=False,
) -> float:
    mse = torch.nn.MSELoss()
    sum_loss = 0
    for image, depth in tqdm(data_loader):
        assert image.shape[0] == 1
        if task_specific:  # DepthPGD
            adv_img = attack.run(
                model=encoder_decoder,
                image=image.to(device),
                label=depth.to(device),
                device=device,
                task_cfg=task_cfg,
                return_step_by_step=False,
            )
        else:  # EmbedAttack
            adv_img, _ = attack.run(
                images=image.to(device),
                model=encoder_decoder,
            )
        with torch.no_grad():
            preds = encoder_decoder.whole_inference(
                adv_img, None, rescale=True
            ).squeeze(1)
            rmse = torch.sqrt(mse(preds.cpu(), depth.cpu())).item()
            sum_loss += rmse
    return sum_loss / len(data_loader)

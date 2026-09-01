import numpy as np
import torch
from itertools import product
from src.models import Encoder, Segmenter

from typing import Callable, Optional


def intersection_union(imPred, imLab, numClass) -> tuple[np.ndarray, np.ndarray]:
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    area_intersection, _ = np.histogram(
        intersection, bins=numClass, range=(1, numClass)
    )

    # Compute area union:
    area_pred, _ = np.histogram(imPred, bins=numClass, range=(1, numClass))
    area_lab, _ = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return area_intersection, area_union


def correct_total(imPred, imLab, numClass) -> tuple[np.ndarray, np.ndarray]:
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    class_total = np.bincount(imPred.flatten(), minlength=numClass)[1:]
    class_correct = np.bincount(imLab[imPred == imLab].flatten(), minlength=numClass)[
        1:
    ]

    return class_correct, class_total


def get_macc_miou(
    preds: np.ndarray | list,
    labels: np.ndarray | list,
    num_classes: int,
) -> dict:
    intersections, unions, corrects, totals = [], [], [], []

    for pred, label in zip(preds, labels):
        i, u = intersection_union(pred, label, num_classes)
        c, t = correct_total(pred, label, num_classes)
        intersections.append(i)
        unions.append(u)
        corrects.append(c)
        totals.append(t)

    return {
        "mean_accuracy": np.nanmean(np.sum(corrects, axis=0) / np.sum(totals, axis=0)),
        "mean_iou": np.nanmean(
            np.sum(intersections, axis=0) / np.sum(unions, axis=0), axis=0
        ),
    }


def sliding_window_inference(
    model: torch.nn.Module,
    num_classes: int,
    input_data: torch.Tensor,
    crop_size,
    stride,
    batch_size: int,
    encoder: Optional[torch.nn.Module] = None,
    return_logits: bool = False,
) -> torch.Tensor:
    softmax = torch.nn.Softmax(dim=1)

    _, _, height, width = input_data.size()
    output = torch.zeros((1, num_classes, height, width)).to(input_data.device)
    freq = output.clone()

    xs = np.unique(
        [
            x if x + crop_size[1] <= width else width - crop_size[1]
            for x in range(0, width, stride[1])
        ]
    )
    ys = np.unique(
        [
            y if y + crop_size[0] <= height else height - crop_size[0]
            for y in range(0, height, stride[0])
        ]
    )

    crops = []
    for x, y in product(xs, ys):
        crops.append(input_data[:, :, y : y + crop_size[0], x : x + crop_size[1]])

    if return_logits:
        preds = []
        for i in range(0, len(crops), batch_size):  # this should be faster :)
            batch = torch.cat(crops[i:])

            pred = model(batch)
            preds += [softmax(p.unsqueeze(0)) for p in pred]

        for i, (x, y) in enumerate(product(xs, ys)):
            output[:, :, y : y + crop_size[0], x : x + crop_size[1]] += preds[i]
            freq[:, :, y : y + crop_size[0], x : x + crop_size[1]] += 1

        output /= freq

        return output

    else:
        assert encoder is not None, "Encoder is required for patch embeddings"
        embeddings = []
        for i in range(0, len(crops), batch_size):
            batch = torch.cat(crops[i:])

            embedding = encoder.get_patch_embeddings(batch)
            embeddings += [embedding]

        return torch.cat(embeddings).unsqueeze(0)


def get_patch_embeddings_pgd(
    segmenter: Segmenter,
    encoder: Encoder,
    n_labels: int,
    device,
    model_cfg,
    task_cfg,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def _internal_get(image: torch.Tensor) -> torch.Tensor:
        return sliding_window_inference(
            segmenter,
            encoder=encoder,
            num_classes=n_labels,
            input_data=image.to(device),
            crop_size=(
                encoder.image_size,
                encoder.image_size,
            ),
            stride=(model_cfg.stride_size, model_cfg.stride_size),
            batch_size=task_cfg.batch_size,
            return_logits=False,
        )

    return _internal_get

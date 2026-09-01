from src.evaluation.utils import get_macc_miou, sliding_window_inference
from src.models import Segmenter, LinearClassifier, DepthEncoderDecoder
from torch.utils.data import DataLoader

import torch
from tqdm import tqdm


def evaluate_accuracy(
    classifier: LinearClassifier, loader: DataLoader, device
) -> float:
    classifier.clf.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = classifier(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    classifier.clf.train()
    return correct / total


def evaluate_segmenter_accuracy_miou(
    segmenter: Segmenter,
    dataloader: DataLoader,
    device,
    model_cfg: dict,
    task_cfg: dict,
) -> float:
    segmenter.head.eval()
    img_size = segmenter.encoder.image_size
    num_classes = segmenter.head.num_labels
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, label = batch
            inputs = inputs.to(device)
            pred = (
                sliding_window_inference(
                    model=segmenter,
                    num_classes=num_classes,
                    input_data=inputs,
                    crop_size=(img_size, img_size),
                    stride=(model_cfg.stride_size, model_cfg.stride_size),
                    batch_size=task_cfg.batch_size,
                    return_logits=True,
                )
                .cpu()
                .argmax(dim=1)[0]
            )
            predictions.append(pred)
            labels.append(label)

    segmenter.head.train()
    return get_macc_miou(
        preds=predictions, labels=labels, num_classes=segmenter.head.num_labels
    )


def evaluate_depth_estimation(
    encoder_decoder: DepthEncoderDecoder,
    dataLoader: DataLoader,
    device: torch.device,
):
    sum_loss = 0
    model = encoder_decoder.decode_head
    model.encoder.eval()
    model.encoder.freeze()
    mse = torch.nn.MSELoss()
    with torch.no_grad():
        for image, depth in tqdm(dataLoader, total=len(dataLoader)):
            assert image.shape[0] == 1
            image = image.to(device)
            depth = depth.cpu()
            valid_mask = depth > 0
            max_depth = encoder_decoder.decode_head.max_depth
            if max_depth is not None:
                valid_mask = torch.logical_and(depth > 0, depth <= max_depth)
            depth = depth[valid_mask]
            pred = encoder_decoder.whole_inference(image, None, rescale=True)
            pred = pred.squeeze(1).detach().cpu()
            pred = pred[valid_mask]
            rmse = torch.sqrt(mse(pred, depth)).item()
            sum_loss += rmse

    return sum_loss / len(dataLoader)

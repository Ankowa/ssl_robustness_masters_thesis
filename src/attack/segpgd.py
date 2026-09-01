from src.attack.base import PGD
from src.evaluation.utils import sliding_window_inference
from src.models import Segmenter

import torch


# adapted based on https://github.com/asif-hanif/segpgd/blob/main/segpgd.py
class SegPGD(PGD):
    name = "SegPGD"

    def run(
        self,
        model: Segmenter,
        image: torch.Tensor,
        label: torch.Tensor,
        model_cfg: dict,
        task_cfg: dict,
        device=None,
        targeted=False,
    ):
        # model for volumetric image segmentation
        # images: [B,C,H,W]. B=BatchSize, C=Number-of-Channels,  H=Height,  W=Width
        # label: [B,1,H,W] (in integer form)

        img_size = model.encoder.image_size
        num_classes = model.head.num_labels

        image = self._denormalize(image).clone().detach().to(device)  #  [B,C,H,W]
        label = label.clone().detach().to(device)  #  [B,H,W]
        adv_image = image.clone().detach()

        # starting at a uniformly random point
        self.delta = torch.empty_like(adv_image).uniform_(
            -self.eps_budget, self.eps_budget
        )

        self._clamp_delta(image)

        adv_image = adv_image + self.delta

        for i in range(self.num_steps):
            adv_image.requires_grad = True
            # adv_image[B,C,H,W] --> adv_logits[B,NumClass,H,W]
            adv_logits = sliding_window_inference(
                model,
                num_classes=num_classes,
                input_data=self._normalize(adv_image),
                crop_size=(img_size, img_size),
                stride=(model_cfg.stride_size, model_cfg.stride_size),
                batch_size=task_cfg.batch_size,
                return_logits=True,
            )
            # [B,NumClass,H,W,D] --> [B,H,W]
            pred_label = torch.argmax(adv_logits, dim=1)
            # correctly classified voxels  [B,H,W]
            correct_voxels = label == pred_label
            # wrongly classified voxels    [B,H,W]
            wrong_voxels = label != pred_label
            # [B,NumClass,H,W] -->  [NumClass,B,H,W]
            adv_logits = adv_logits.permute(1, 0, 2, 3)
            # calculate loss
            loss_correct = self.loss(
                adv_logits[:, correct_voxels].permute(1, 0),
                label[correct_voxels],
            )
            loss_wrong = self.loss(
                adv_logits[:, wrong_voxels].permute(1, 0),
                label[wrong_voxels],
            )
            lmbda = i / (2 * self.num_steps)
            loss = (1 - lmbda) * loss_correct + lmbda * loss_wrong
            if targeted:
                loss = -1 * loss

            # update adversarial image
            grad = torch.autograd.grad(
                loss, adv_image, retain_graph=False, create_graph=False
            )[0]
            adv_image = adv_image.detach() + self.lr * grad.sign()
            self.delta = adv_image - image
            self._clamp_delta(image)
            adv_image = image + self.delta
        return adv_image.detach()

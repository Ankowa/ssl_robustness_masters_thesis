from src.attack.base import PGD

from src.models.downstream_models.depther.encoder_decoder import DepthEncoderDecoder

from src.evaluation.depth_losses.sigloss import SigLoss
from src.evaluation.depth_losses.gradientloss import GradientLoss

import torch


class DepthPGD(PGD):
    name = "DepthPGD"

    def run(
        self,
        model: DepthEncoderDecoder,
        image: torch.Tensor,
        label: torch.Tensor,
        task_cfg: dict,
        device=None,
    ):
        image = self._denormalize(image).clone().detach().to(device)  #  [B,C,H,W]
        target = label.clone().detach().to(device)  #  [B,H,W]
        adv_image = image.clone().detach()

        # starting at a uniformly random point
        self.delta = torch.empty_like(adv_image).uniform_(
            -self.eps_budget, self.eps_budget
        )

        self._clamp_delta(image)

        adv_image = adv_image + self.delta

        sig_loss = SigLoss(
            valid_mask=True,
            loss_weight=task_cfg.sig_loss_weight,
            warm_up=True,
            loss_name="loss_depth",
        )
        grad_loss = GradientLoss(
            valid_mask=True,
            loss_weight=task_cfg.grad_loss_weight,
            loss_name="loss_grad",
        )

        for i in range(self.num_steps):
            adv_image.requires_grad = True
            adv_preds = model.whole_inference(
                self._normalize(adv_image), None, rescale=True
            ).squeeze(1)
            loss = sig_loss(adv_preds, target) + grad_loss(adv_preds, target)
            grad = torch.autograd.grad(
                loss, adv_image, retain_graph=False, create_graph=False
            )[0]
            adv_image = adv_image.detach() + self.lr * grad.sign()
            self.delta = adv_image - image
            self._clamp_delta(image)
            adv_image = image + self.delta
        return adv_image.detach()

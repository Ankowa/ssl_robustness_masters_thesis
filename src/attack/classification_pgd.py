import torch

from src.attack.base import PGD
from src.models import LinearClassifier


class ClassificationPGD(PGD):
    name = "clf_pgd"

    def __init__(self, num_steps, lr, eps_budget, loss=None, mean=None, std=None):
        super().__init__(self.loss, num_steps, lr, eps_budget, mean, std)

    def _init_delta_optim(self, images: torch.Tensor):
        self.delta = torch.zeros_like(images).uniform_(
            -self.eps_budget, self.eps_budget
        )
        self.delta = self.delta.detach().requires_grad_(True)

    def _optimize_delta(self):
        self.delta.data += torch.sign(self.delta.grad) * self.lr

    def run(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        model: LinearClassifier,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = self._denormalize(images)
        self._init_delta_optim(images)

        for _ in range(self.num_steps):
            logits = model(self._normalize(images + self.delta))
            with torch.no_grad():
                div = torch.maximum(logits, torch.ones_like(logits)).max(
                    dim=1, keepdim=True
                )[0]
            self.loss(logits / div, labels).backward()

            self._optimize_delta()
            self._clamp_delta(images)

            self.delta.grad.zero_()

        adv_images = self._normalize(images + self.delta)
        success = torch.argmax(model(adv_images), dim=1) != labels
        return adv_images.detach(), success.detach()

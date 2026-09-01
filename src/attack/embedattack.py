from src.attack.base import PGD
import torch


class EmbedAttack(PGD):
    name = "embedattack"

    def _init_delta_optim(self, images: torch.Tensor):
        self.delta = torch.zeros_like(images).uniform_(
            -self.eps_budget, self.eps_budget
        )
        self.delta = self.delta.detach().requires_grad_(True)

    def _optimize_delta(self):
        self.delta.data += torch.sign(self.delta.grad) * self.lr

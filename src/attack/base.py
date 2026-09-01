import numpy as np
import torch
from src.models import Encoder

from typing import Callable
from torchvision import transforms


class Attack:
    name: str

    def __init__(
        self,
        loss: Callable,
        num_steps: int,
        lr: float,
        eps_budget: float,
        mean: tuple = None,
        std: tuple = None,
    ):
        self.loss = loss
        self.num_steps = num_steps
        self.lr = self._cast_to_float(lr)
        self.eps_budget = self._cast_to_float(eps_budget)
        self.left_boundary = 0
        self.right_boundary = 1
        self.mean = mean
        self.std = std

        self._normalize = transforms.Normalize(mean, std)
        self._denormalize = transforms.Normalize(
            -np.array(mean) / np.array(std), 1 / np.array(std)
        )

    def run(self, images, model, **kwargs):
        raise NotImplementedError

    def _cast_to_float(self, val) -> float:
        if not isinstance(val, float):
            try:
                return float(eval(val))
            except NameError:
                raise ValueError(
                    f"Value should be a float or a string that can be evaluated to a float (e.g. 1/255), got {val}"
                )
        else:
            return val


class PGD(Attack):
    name: str
    loss: Callable
    num_steps: int
    lr: float
    eps_budget: float
    optimizer: torch.optim.Optimizer = None
    delta: torch.Tensor = None

    def _init_delta_optim(self, images: torch.Tensor):
        raise NotImplementedError

    @torch.no_grad()
    def _clamp_delta(self, images):
        # makes sure the delta is within budget and we don't get out of bounds of
        # [lb; rb] for images + delta
        self.delta.data = (
            torch.clamp_(
                images + self.delta.data,
                min=self.left_boundary,
                max=self.right_boundary,
            )
            - images
        )
        self.delta.data = torch.clamp_(
            self.delta.data, min=-self.eps_budget, max=self.eps_budget
        )

    def _optimize_delta(self):
        raise NotImplementedError

    def run(
        self,
        images: torch.Tensor,
        model: Encoder,
        precomputed_original_representations: torch.Tensor = None,
        get_representation: Callable = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = self._denormalize(images)
        if get_representation is None:
            get_representation = model.get_representation
        if precomputed_original_representations is None:
            with torch.no_grad():
                original_representations = get_representation(self._normalize(images))
        else:
            original_representations = precomputed_original_representations

        for idx in range(self.num_steps):
            adv_representations = get_representation(
                self._normalize(images + self.delta)
            )
            self.loss(adv_representations, original_representations).backward()

            self._optimize_delta()
            self._clamp_delta(images)

            if self.optimizer is not None:
                self.optimizer.zero_grad()
            else:
                self.delta.grad.zero_()

        adv_representations = get_representation(self._normalize(images + self.delta))
        adv_images: torch.Tensor = self._normalize(images + self.delta)
        return adv_images.detach(), adv_representations.detach()

import os, torch
from src.models.ssl_models.ssl import Encoder


class DownstreamClassifier(torch.nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        dataset_name: str,
        lr: int,
        epochs: int,
        num_classes: int,
        device="cuda",
        is_train: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_name = encoder.name
        self.dataset_name = dataset_name
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.num_classes = num_classes
        self.is_train = is_train


class LinearClassifier(DownstreamClassifier):
    def __init__(
        self,
        encoder: Encoder,
        dataset_name: str,
        lr: int,
        epochs: int,
        num_classes: int,
        device="cuda",
        is_train: bool = False,
    ):
        super().__init__(
            encoder, dataset_name, lr, epochs, num_classes, device, is_train
        )
        self.encoder.eval()
        self.clf = torch.nn.Linear(encoder.feature_dim, num_classes)
        self.clf.to(self.device)

    def get_name(self) -> str:
        return f"tmp/classifiers/linear_{self.encoder_name}_{self.dataset_name}_{self.lr}_{self.epochs}.pt"

    def load_clf(self):
        self.clf.load_state_dict(
            torch.load(
                self.get_name(),
                map_location=torch.device("cpu"),
            )
        )

    def save_clf(self):
        if not os.path.exists("tmp/classifiers"):
            os.mkdir("tmp/classifiers")
        path = self.get_name()
        torch.save(self.clf.state_dict(), path)

    def exists_clf(self):
        return os.path.exists(self.get_name())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_r = self.encoder.get_representation(x)
        y = self.clf(img_r)
        return y

    def clf_params(self):
        return self.clf.parameters()

from src.attack.base import Attack, PGD
from src.attack.embedattack import EmbedAttack
from src.attack.classification_pgd import ClassificationPGD
from src.attack.segpgd import SegPGD
from src.attack.depthpgd import DepthPGD

pgd_dict: dict[str, PGD] = {
    "embedattack": EmbedAttack,
    "clf_pgd": ClassificationPGD,
    "segpgd": SegPGD,
    "depthpgd": DepthPGD,
}

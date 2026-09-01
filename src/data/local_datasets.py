from pathlib import Path
from typing import Callable, Optional

import torch
import torchvision
from datasets import load_dataset, load_from_disk
import numpy as np


class HFDataset:
    name: str
    hg_name: str
    dataset_class: torchvision.datasets.VisionDataset
    image: Optional[str] = "image"
    label: Optional[str] = "label"
    train_split_name: Optional[str] = "train"
    test_split_name: Optional[str] = "test"
    classes_cnt: Optional[int] = None

    def __init__(self, root: str):
        self.root = root
        self.data = None

    def load_dataset(self, split: str) -> None:
        assert split in [
            "train",
            "test",
        ], f'Only split="train" and split="test" supported. Provided: split={split}'
        split = self.train_split_name if split == "train" else self.test_split_name
        self.data = load_dataset(self.hg_name, split=split, cache_dir=self.root)


class STL10(HFDataset):
    name = "stl10"
    hg_name = "jxie/stl10"
    classes_cnt = 10


class CIFAR10(HFDataset):
    name = "cifar10"
    hg_name = "uoft-cs/cifar10"
    image = "img"
    classes_cnt = 10


class CIFAR100(HFDataset):
    name = "cifar100"
    image = "img"
    label = "fine_label"
    hg_name = "uoft-cs/cifar100"
    classes_cnt = 100


class ADE20k(HFDataset):
    name = "ade20k"
    hg_name = None
    classes_cnt = 151
    test_split_name = "val"
    id2label = {
        0: "background",
        1: "wall",
        2: "building, edifice",
        3: "sky",
        4: "floor, flooring",
        5: "tree",
        6: "ceiling",
        7: "road, route",
        8: "bed ",
        9: "windowpane, window ",
        10: "grass",
        11: "cabinet",
        12: "sidewalk, pavement",
        13: "person, individual, someone, somebody, mortal, soul",
        14: "earth, ground",
        15: "door, double door",
        16: "table",
        17: "mountain, mount",
        18: "plant, flora, plant life",
        19: "curtain, drape, drapery, mantle, pall",
        20: "chair",
        21: "car, auto, automobile, machine, motorcar",
        22: "water",
        23: "painting, picture",
        24: "sofa, couch, lounge",
        25: "shelf",
        26: "house",
        27: "sea",
        28: "mirror",
        29: "rug, carpet, carpeting",
        30: "field",
        31: "armchair",
        32: "seat",
        33: "fence, fencing",
        34: "desk",
        35: "rock, stone",
        36: "wardrobe, closet, press",
        37: "lamp",
        38: "bathtub, bathing tub, bath, tub",
        39: "railing, rail",
        40: "cushion",
        41: "base, pedestal, stand",
        42: "box",
        43: "column, pillar",
        44: "signboard, sign",
        45: "chest of drawers, chest, bureau, dresser",
        46: "counter",
        47: "sand",
        48: "sink",
        49: "skyscraper",
        50: "fireplace, hearth, open fireplace",
        51: "refrigerator, icebox",
        52: "grandstand, covered stand",
        53: "path",
        54: "stairs, steps",
        55: "runway",
        56: "case, display case, showcase, vitrine",
        57: "pool table, billiard table, snooker table",
        58: "pillow",
        59: "screen door, screen",
        60: "stairway, staircase",
        61: "river",
        62: "bridge, span",
        63: "bookcase",
        64: "blind, screen",
        65: "coffee table, cocktail table",
        66: "toilet, can, commode, crapper, pot, potty, stool, throne",
        67: "flower",
        68: "book",
        69: "hill",
        70: "bench",
        71: "countertop",
        72: "stove, kitchen stove, range, kitchen range, cooking stove",
        73: "palm, palm tree",
        74: "kitchen island",
        75: "computer, computing machine, computing device, data processor, electronic computer, information processing system",
        76: "swivel chair",
        77: "boat",
        78: "bar",
        79: "arcade machine",
        80: "hovel, hut, hutch, shack, shanty",
        81: "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle",
        82: "towel",
        83: "light, light source",
        84: "truck, motortruck",
        85: "tower",
        86: "chandelier, pendant, pendent",
        87: "awning, sunshade, sunblind",
        88: "streetlight, street lamp",
        89: "booth, cubicle, stall, kiosk",
        90: "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box",
        91: "airplane, aeroplane, plane",
        92: "dirt track",
        93: "apparel, wearing apparel, dress, clothes",
        94: "pole",
        95: "land, ground, soil",
        96: "bannister, banister, balustrade, balusters, handrail",
        97: "escalator, moving staircase, moving stairway",
        98: "ottoman, pouf, pouffe, puff, hassock",
        99: "bottle",
        100: "buffet, counter, sideboard",
        101: "poster, posting, placard, notice, bill, card",
        102: "stage",
        103: "van",
        104: "ship",
        105: "fountain",
        106: "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
        107: "canopy",
        108: "washer, automatic washer, washing machine",
        109: "plaything, toy",
        110: "swimming pool, swimming bath, natatorium",
        111: "stool",
        112: "barrel, cask",
        113: "basket, handbasket",
        114: "waterfall, falls",
        115: "tent, collapsible shelter",
        116: "bag",
        117: "minibike, motorbike",
        118: "cradle",
        119: "oven",
        120: "ball",
        121: "food, solid food",
        122: "step, stair",
        123: "tank, storage tank",
        124: "trade name, brand name, brand, marque",
        125: "microwave, microwave oven",
        126: "pot, flowerpot",
        127: "animal, animate being, beast, brute, creature, fauna",
        128: "bicycle, bike, wheel, cycle ",
        129: "lake",
        130: "dishwasher, dish washer, dishwashing machine",
        131: "screen, silver screen, projection screen",
        132: "blanket, cover",
        133: "sculpture",
        134: "hood, exhaust hood",
        135: "sconce",
        136: "vase",
        137: "traffic light, traffic signal, stoplight",
        138: "tray",
        139: "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
        140: "fan",
        141: "pier, wharf, wharfage, dock",
        142: "crt screen",
        143: "plate",
        144: "monitor, monitoring device",
        145: "bulletin board, notice board",
        146: "shower",
        147: "radiator",
        148: "glass, drinking glass",
        149: "clock",
        150: "flag",
    }

    def load_dataset(self, split: str) -> None:
        assert split in [
            "train",
            "test",
        ], f'Only split="train" and split="test" supported. Provided: split={split}'
        split = self.train_split_name if split == "train" else self.test_split_name
        self.data = load_from_disk("tmp/data/ade20k_hg")[split]


class CityScapes(HFDataset):
    name = "cityscapes"
    hg_name = None
    classes_cnt = 20
    test_split_name = "val"
    id2label = {
        0: "background/unknown/unlabeled",
        1: "road",
        2: "sidewalk",
        3: "building",
        4: "wall",
        5: "fence",
        6: "pole",
        7: "traffic light",
        8: "traffic sign",
        9: "vegetation",
        10: "terrain",
        11: "sky",
        12: "person",
        13: "raider",
        14: "car",
        15: "truck",
        16: "bus",
        17: "train",
        18: "motorcycle",
        19: "bicycle",
    }

    def load_dataset(self, split: str) -> None:
        assert split in [
            "train",
            "test",
        ], f'Only split="train" and split="test" supported. Provided: split={split}'
        split = self.train_split_name if split == "train" else self.test_split_name
        self.data = load_from_disk("tmp/data/cityscapes_hg")[split]


class PascalVOC2012(HFDataset):
    name = "pascal_voc_2012"
    hg_name = None
    classes_cnt = 21
    test_split_name = "val"
    id2label = {
        0: "background/void",
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car ",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "potted plant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tv/monitor",
    }

    def load_dataset(self, split: str) -> None:
        assert split in [
            "train",
            "test",
        ], f'Only split="train" and split="test" supported. Provided: split={split}'
        split = self.train_split_name if split == "train" else self.test_split_name
        self.data = load_from_disk("tmp/data/pascal_voc_hg")[split]


class NYUDepthV2(HFDataset):
    name = "nyu_depth_v2"
    hg_name = "sayakpaul/nyu_depth_v2"
    classes_cnt = None
    test_split_name = "validation"
    label = "depth_map"
    depth_scale = 1


class HFClassificationDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        dataset: HFDataset,
        split: str,
        root: str,
        transform: Callable = None,
    ):
        self.root = Path(root)
        self.input_transform = transform
        self.dataset = dataset
        self.dataset.load_dataset(split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = (
            self.dataset.data[index][self.dataset.image],
            self.dataset.data[index][self.dataset.label],
        )
        if self.input_transform is not None:
            image = self.input_transform(image.convert("RGB"))
        return image, label

    def __len__(self):
        return len(self.dataset.data)


class HFSegmentationDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        dataset: HFDataset,
        split: str,
        root: str,
        transform: Callable = None,
    ):
        self.root = Path(root)
        self.transform = transform.transform
        self.dataset = dataset
        self.dataset.load_dataset(split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = (
            np.array(self.dataset.data[index][self.dataset.image]),
            np.array(self.dataset.data[index][self.dataset.label]),
        )
        if self.transform is not None:
            try:
                transformed = self.transform(image=image, mask=label)
            except:
                print(image.shape, label.shape, index)
                label = label.T  # for foodseg103 it happens only for some images
                print("Transposing label due to shape inconsistency, index:", index)
                transformed = self.transform(image=image, mask=label)

            image, label = torch.tensor(transformed["image"]), torch.LongTensor(
                transformed["mask"]
            )
            # convert to C, H, W
            image = image.permute(2, 0, 1)
        return image, label

    def __len__(self):
        return len(self.dataset.data)


class HFDepthDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        dataset: HFDataset,
        split: str,
        root: str,
        transform: Callable = None,
    ):
        self.root = Path(root)
        self.transform = transform.transform
        self.dataset = dataset
        self.dataset.load_dataset(split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = (
            np.array(self.dataset.data[index][self.dataset.image]),
            np.array(self.dataset.data[index][self.dataset.label]),
        )
        label = label / self.dataset.depth_scale
        if self.transform is not None:
            try:
                transformed = self.transform(image=image, mask=label)
            except:
                print(image.shape, label.shape, index)
                label = label.T
                print("Transposing label due to shape inconsistency, index:", index)
                transformed = self.transform(image=image, mask=label)

            image, label = torch.tensor(transformed["image"]), torch.tensor(
                transformed["mask"]
            )
            # convert to C, H, W
            image = image.permute(2, 0, 1)
        return image, label

    def __len__(self):
        return len(self.dataset.data)


dataset_dict: dict[str, HFDataset] = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "ade20k": ADE20k,
    "stl10": STL10,
    "cityscapes": CityScapes,
    "pascal_voc_2012": PascalVOC2012,
    "nyu_depth_v2": NYUDepthV2,
}

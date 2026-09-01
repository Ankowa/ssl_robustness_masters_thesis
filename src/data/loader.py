# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# adapted from https://github.com/pantheon5100/DeACL/blob/dc0807e0b2b133fec3c9a3ec2dca6f3a2527cb5e
# /solo/utils/pretrain_dataloader_AdvTraining.py

import os
from pathlib import Path
from typing import Optional, Union, Callable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import Dataset
from src.data.transforms import prepare_transform, prepare_n_crop_transform
from src.data.local_datasets import (
    dataset_dict,
    HFClassificationDataset,
    HFSegmentationDataset,
    HFDepthDataset,
)


def prepare_datasets(
    dataset: str,
    transform: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    sub_dir: Optional[Union[str, Path]] = None,
    split: str = "train",
) -> Dataset:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        data_dir (Optional[Union[str, Path]], optional): the directory to load data from.
            Defaults to None.
        sub_dir (Optional[Union[str, Path]], optional): data directory to be appended to
        data_dir. Defaults to None.
        split (str, optional): the split of the dataset. Defaults to "train".

    Returns:
        Dataset: the desired dataset with transformations.
    """

    if data_dir is None:
        sandbox_folder = Path(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        data_dir = sandbox_folder / "datasets"

    if sub_dir is not None:
        data_dir = Path(f"{data_dir}/{sub_dir}")

    if dataset in ("ade20k", "cityscapes", "pascal_voc_2012"):
        hg_dataset = dataset_dict[dataset](
            root=data_dir,
        )
        dataset = HFSegmentationDataset(
            dataset=hg_dataset,
            split=split,
            root=data_dir,
            transform=transform,
        )
    elif dataset == "nyu_depth_v2":
        hg_dataset = dataset_dict[dataset](
            root=data_dir,
        )
        dataset = HFDepthDataset(
            dataset=hg_dataset,
            split=split,
            root=data_dir,
            transform=transform,
        )
    elif dataset in dataset_dict.keys():
        hg_dataset = dataset_dict[dataset](
            root=data_dir,
        )
        dataset = HFClassificationDataset(
            dataset=hg_dataset,
            split=split,
            root=data_dir,
            transform=transform,
        )

    else:
        raise Exception(f"Unknown dataset {dataset=}")
    return dataset


def prepare_dataloader(
    dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader


def get_dataloader(
    dataset: str,
    data_dir: Optional[Union[str, Path]] = None,
    sub_dir: Optional[Union[str, Path]] = None,
    split: Optional[str] = "train",
    batch_size: Optional[int] = 64,
    num_workers: Optional[int] = 4,
    two_views: Optional[bool] = False,
    augment: Optional[bool] = True,
    crop_size: Optional[int] = 224,
    normalize: Optional[bool] = True,
) -> DataLoader:
    transform = prepare_transform(
        dataset, augment=augment, crop_size=crop_size, split=split, normalize=normalize
    )
    if two_views:
        transform = prepare_n_crop_transform([transform], [2])
    dataset = prepare_datasets(dataset, transform, data_dir, sub_dir, split=split)
    dataloader = prepare_dataloader(dataset, batch_size, num_workers)
    return dataloader

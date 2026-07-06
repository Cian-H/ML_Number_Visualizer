from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import QMNIST
from torchvision.transforms import ToTensor

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets/"


def create_dataset(
    split: tuple[float, float, float] = (0.7, 0.1, 0.2),
    dataset=QMNIST,
    **kwargs,
):
    ds = dataset(str(DATASET_DIR), download=True, transform=ToTensor())

    shuffle = kwargs.pop("shuffle", False)
    shuffle_train = kwargs.pop("shuffle_train", False)

    num_workers = kwargs.get("num_workers", 0)
    kwargs.setdefault("num_workers", 0)
    kwargs.setdefault("pin_memory", num_workers > 0)
    if num_workers > 0:
        kwargs.setdefault("prefetch_factor", 8)
        kwargs.setdefault("persistent_workers", True)

    to_shuffle = (shuffle or shuffle_train, shuffle, shuffle)
    train, val, test = (
        DataLoader(i, shuffle=s, **kwargs)
        for i, s in zip(random_split(ds, split), to_shuffle, strict=True)
    )
    return train, val, test


def collate(batch):
    x, y = zip(*batch, strict=True)
    x = torch.stack(x)
    y = torch.tensor(y)
    return x, y


# This is just a quick, lazy way to ensure all models are trained on the same dataset
@lru_cache(maxsize=1)
def get_dataset():
    from torchvision.datasets import QMNIST

    from .dataloader import create_dataset

    return create_dataset(
        dataset=QMNIST,
        collate_fn=collate,
        batch_size=256,
        shuffle_train=True,
        num_workers=0,
    )

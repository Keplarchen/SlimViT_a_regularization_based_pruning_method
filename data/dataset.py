from transform import *
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import random_split, Dataset

def dataset_split(full_dataset: Dataset,
                  val_ratio: float=0.75) -> tuple[Dataset, Dataset]:
    """

    :param full_dataset:
    :param val_ratio:
    :return:
    """
    train_size = int(len(full_dataset) * val_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    return train_dataset, val_dataset

def cifar_dataset(target_dataset: str,
                  root: str="cifar100",
                  download: bool=True,
                  val_dataset: bool=False,
                  val_ratio: float=0.75) -> tuple[Dataset, Dataset, Dataset]:
    if target_dataset == "cifar10":
        full_dataset = CIFAR10(
            root=root,
            train=True,
            download=download,
            transform=CIFAR10_transform
        )
        test_dataset = CIFAR10(
            root=root,
            train=False,
            download=download,
            transform=CIFAR10_transform
        )
    elif target_dataset == "cifar100":
        full_dataset = CIFAR100(
            root=root,
            train=True,
            download=download,
            transform=CIFAR100_transform
        )
        test_dataset = CIFAR100(
            root=root,
            train=False,
            download=download,
            transform=CIFAR100_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {target_dataset}")
    train_dataset, val_dataset = dataset_split(full_dataset, val_ratio if val_dataset else 0)
    return train_dataset, val_dataset, test_dataset

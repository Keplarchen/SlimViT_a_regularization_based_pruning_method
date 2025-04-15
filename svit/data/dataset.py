from svit import config
from svit.data.transform import CIFAR_transform

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import random_split, Dataset

def dataset_split(full_dataset: Dataset,
                  val_ratio: float=config["data"]["val_ratio"]) -> tuple[Dataset, Dataset]:
    """

    :param full_dataset:
    :param val_ratio:
    :return:
    """
    train_size = int(len(full_dataset) * val_ratio)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    return train_dataset, val_dataset

def cifar_dataset(target_dataset: str=config["data"]["train_dataset"],
                  root: str="cifar100",
                  download: bool=True,
                  val_dataset: bool=config["data"]["val_dataset"],
                  val_ratio: float=config["data"]["val_ratio"]) -> tuple[Dataset, Dataset, Dataset]:
    if target_dataset == "cifar10":
        full_dataset = CIFAR10(
            root=root,
            train=True,
            download=download,
            transform=CIFAR_transform()
        )
        test_dataset = CIFAR10(
            root=root,
            train=False,
            download=download,
            transform=CIFAR_transform()
        )
    elif target_dataset == "cifar100":
        full_dataset = CIFAR100(
            root=root,
            train=True,
            download=download,
            transform=CIFAR_transform()
        )
        test_dataset = CIFAR100(
            root=root,
            train=False,
            download=download,
            transform=CIFAR_transform()
        )
    else:
        raise ValueError(f"Unknown dataset: {target_dataset}")
    if val_dataset:
        train_dataset, val_dataset = dataset_split(full_dataset, val_ratio)
    else:
        train_dataset, val_dataset = full_dataset, None
    return train_dataset, val_dataset, test_dataset

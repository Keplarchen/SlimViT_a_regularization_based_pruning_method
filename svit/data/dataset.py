from svit.data.transform import CIFAR_transform, ImageNet_transform

from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import random_split, Dataset

def dataset_split(full_dataset: Dataset,
                  train_ratio: float) -> tuple[Dataset, Dataset]:
    """

    :param full_dataset:
    :param train_ratio:
    :return:
    """
    train_size = int(len(full_dataset) * train_ratio)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(len(train_dataset), len(val_dataset))
    return train_dataset, val_dataset

def cifar10_dataset(train_root: str,
                    test_root: str,
                    val_dataset: bool,
                    train_ratio: float,
                    resize_size: int) -> tuple[Dataset, Dataset, Dataset]:
    """

    :param train_root:
    :param test_root:
    :param download:
    :param val_dataset:
    :param train_ratio:
    :param resize_size:
    :return:
    """
    full_dataset = CIFAR10(
        root=train_root,
        train=True,
        download=True,
        transform=CIFAR_transform("cifar10", resize_size)
    )
    test_dataset = CIFAR10(
        root=test_root,
        train=False,
        download=True,
        transform=CIFAR_transform("cifar10", resize_size)
    )
    if val_dataset:
        train_dataset, val_dataset = dataset_split(full_dataset, train_ratio)
    else:
        train_dataset, val_dataset = full_dataset, None
    return train_dataset, val_dataset, test_dataset

def cifar100_dataset(train_root: str,
                    test_root: str,
                    val_dataset: bool,
                    train_ratio: float,
                    resize_size: int) -> tuple[Dataset, Dataset, Dataset]:
    """

    :param train_root:
    :param test_root:
    :param val_dataset:
    :param train_ratio:
    :param resize_size:
    :return:
    """
    full_dataset = CIFAR100(
        root=train_root,
        train=True,
        download=True,
        transform=CIFAR_transform("cifar100", resize_size)
    )
    test_dataset = CIFAR100(
        root=test_root,
        train=False,
        download=True,
        transform=CIFAR_transform("cifar100", resize_size)
    )
    if val_dataset:
        train_dataset, val_dataset = dataset_split(full_dataset, train_ratio)
    else:
        train_dataset, val_dataset = full_dataset, None
    return train_dataset, val_dataset, test_dataset

def imagenette_dataset(train_root: str,
                       test_root: str,
                       val_dataset: bool,
                       train_ratio: float,
                       resize_size: int) -> tuple[Dataset, Dataset, Dataset]:
    """

    :param train_root:
    :param test_root:
    :param val_dataset:
    :param train_ratio:
    :param resize_size:
    :return:
    """
    imagenette_full = ImageFolder(
        train_root,
        transform=ImageNet_transform("imagenette", resize_size)
    )
    imagenette_test = ImageFolder(
        test_root,
        transform=ImageNet_transform("imagenette", resize_size)
    )
    if val_dataset:
        imagenette_train, imagenette_val = dataset_split(imagenette_full, train_ratio)
    else:
        imagenette_train, imagenette_val = imagenette_full, None
    return imagenette_train, imagenette_val, imagenette_test
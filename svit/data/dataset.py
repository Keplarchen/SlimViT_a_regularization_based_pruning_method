from svit.data.transform import CIFAR_transform

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import random_split, Dataset

def dataset_split(full_dataset: Dataset,
                  train_ratio: float) -> tuple[Dataset, Dataset]:
    """
    Splits a given dataset into training and validation sets based on the specified
    training ratio. The size of the training set is determined by multiplying the
    length of the full dataset with the train ratio. The remainder is assigned to
    the validation set. The split is performed using a random splitting method.

    :param full_dataset: The full dataset to be split.
    :type full_dataset: Dataset
    :param train_ratio: The ratio of the dataset to be allocated for training.
                        Must be a float between 0 and 1.
    :type train_ratio: float
    :return: A tuple containing the training dataset and the validation dataset.
    :rtype: tuple[Dataset, Dataset]
    """
    train_size = int(len(full_dataset) * train_ratio)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(len(train_dataset), len(val_dataset))
    return train_dataset, val_dataset

def cifar_dataset(target_dataset: str,
                  root: str,
                  download: bool,
                  val_dataset: bool,
                  train_ratio: float,
                  resize_size: int) -> tuple[Dataset, Dataset, Dataset]:
    """
    Prepare CIFAR dataset based on the specified parameters. Handles CIFAR-10
    and CIFAR-100 datasets with options for downloading, splitting, and resizing.

    :param target_dataset: Name of the target dataset. Accepted values are
        "cifar10" or "cifar100".
    :param root: Root directory where the dataset should be stored or
        is already present.
    :param download: Whether to download the dataset if it is not already
        present.
    :param val_dataset: Indicates whether the dataset should be split into
        a validation set in addition to training and testing datasets.
    :param train_ratio: Proportion of the dataset to be used for training
        when splitting into training and validation sets. Used if val_dataset
        is True.
    :param resize_size: Target size to which the images will be resized.

    :return: A tuple where the first element is the training dataset, the
        second element is the validation dataset (or None if val_dataset is
        False), and the third element is the testing dataset.
    """
    if target_dataset == "cifar10":
        full_dataset = CIFAR10(
            root=root,
            train=True,
            download=download,
            transform=CIFAR_transform(target_dataset, resize_size)
        )
        test_dataset = CIFAR10(
            root=root,
            train=False,
            download=download,
            transform=CIFAR_transform(target_dataset, resize_size)
        )
    elif target_dataset == "cifar100":
        full_dataset = CIFAR100(
            root=root,
            train=True,
            download=download,
            transform=CIFAR_transform(target_dataset, resize_size)
        )
        test_dataset = CIFAR100(
            root=root,
            train=False,
            download=download,
            transform=CIFAR_transform(target_dataset, resize_size)
        )
    else:
        raise ValueError(f"Unknown dataset: {target_dataset}")
    if val_dataset:
        train_dataset, val_dataset = dataset_split(full_dataset, train_ratio)
    else:
        train_dataset, val_dataset = full_dataset, None
    return train_dataset, val_dataset, test_dataset

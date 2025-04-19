from svit import config
from svit.data.dataset import cifar10_dataset, cifar100_dataset, imagenette_dataset

from torch.utils.data import DataLoader

def get_dataloader(config: dict=config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """

    :param config:
    :return:
    """
    dataset = config["data"]["train_dataset"]
    train_root = config["data"]["train_root"]
    test_root = config["data"]["test_root"]
    val_dataset = config["data"]["val_dataset"]
    train_ratio = config["data"]["train_ratio"]
    batch_size = config["training"]["batch_size"]
    train_shuffle = True
    val_shuffle = False
    resize_size = config["data"]["resize_size"]

    if dataset == "cifar10":
        train_dataset, val_dataset, test_dataset = cifar10_dataset(train_root, test_root, val_dataset, train_ratio, resize_size)
    elif dataset == "cifar100":
        train_dataset, val_dataset, test_dataset = cifar100_dataset(train_root, test_root, val_dataset, train_ratio, resize_size)
    elif dataset == "imagenette":
        train_dataset, val_dataset, test_dataset = imagenette_dataset(train_root, test_root, val_dataset, train_ratio, resize_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=val_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


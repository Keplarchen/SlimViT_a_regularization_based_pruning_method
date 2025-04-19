from svit import config
from svit.data.dataset import cifar10_dataset, cifar100_dataset, imagenette_dataset

from torch.utils.data import DataLoader

def get_dataloader(dataset: str=config["data"]["train_dataset"],
                     train_root: str=config["data"]["train_root"],
                     test_root: str=config["data"]["test_root"],
                     val_dataset: bool=config["data"]["val_dataset"],
                     train_ratio: float=config["data"]["train_ratio"],
                     batch_size: int=config["training"]["batch_size"],
                     train_shuffle: bool=True,
                     val_shuffle: bool=False,
                     resize_size: int=config["data"]["resize_size"]) -> tuple[DataLoader, DataLoader, DataLoader]:

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


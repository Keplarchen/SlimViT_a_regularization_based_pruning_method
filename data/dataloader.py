from code import config
from dataset import cifar_dataset
from torch.utils.data import DataLoader

def cifar_dataloader(dataset: str=config["data"]["train_dataset"],
                     root: str="cifar100",
                     download: bool=True,
                     val_dataset: bool=config["data"]["val_dataset"],
                     val_ratio: float=config["data"]["val_ratio"],
                     batch_size: int=config["train"]["batch_size"],
                     train_shuffle: bool=True,
                     val_shuffle: bool=False) -> tuple[DataLoader, DataLoader, DataLoader]:
    """

    :param dataset:
    :param root:
    :param download:
    :param val_dataset:
    :param val_ratio:
    :param batch_size:
    :param train_shuffle:
    :param val_shuffle:
    :return:
    """
    train_dataset, val_dataset, test_dataset = cifar_dataset(dataset, root, download, val_dataset, val_ratio)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=val_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

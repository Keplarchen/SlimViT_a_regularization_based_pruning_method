from svit import config
from svit.data.dataset import cifar_dataset

from torch.utils.data import DataLoader

def cifar_dataloader(dataset: str=config["data"]["train_dataset"],
                     root: str=config["data"]["train_dataset"],
                     download: bool=True,
                     val_dataset: bool=config["data"]["val_dataset"],
                     train_ratio: float=config["data"]["train_ratio"],
                     batch_size: int=config["training"]["batch_size"],
                     train_shuffle: bool=True,
                     val_shuffle: bool=False,
                     resize_size: int=config["data"]["resize_size"]) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns data loaders for training, validation, and testing datasets
    using the CIFAR dataset. The function provides options to configure dataset
    parameters such as the training-to-validation split ratio, batch sizes, shuffling,
    and resizing of the data. The created dataloaders allow efficient iteration over
    batches of data during training and evaluation processes.

    :param dataset: The name of the dataset to use.
    :param root: The root directory where the dataset is saved or will be downloaded.
    :param download: Indicates whether to download the dataset if it is not already
        present at the specified root path.
    :param val_dataset: Indicates if a validation dataset should be created
        from the training data.
    :param train_ratio: The proportion of the dataset to be used for training (used
        to split into training and validation sets).
    :param batch_size: The number of samples per batch in the dataloader.
    :param train_shuffle: Specifies whether to shuffle the training dataset for
        loading batches.
    :param val_shuffle: Specifies whether to shuffle the validation dataset for
        loading batches.
    :param resize_size: The size to which CIFAR dataset images will be resized.
    :return: A tuple containing the training dataloader, validation dataloader, and
        test dataloader, respectively.
    """
    train_dataset, val_dataset, test_dataset = cifar_dataset(dataset, root, download, val_dataset, train_ratio, resize_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=val_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

from svit import config

from torchvision import transforms

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

def CIFAR_transform(target_dataset: str=config["data"]["train_dataset"],
                    resize_size: int=config["data"]["resize_size"]) -> transforms.Compose:
    """

    :param target_dataset:
    :param resize_size:
    :return:
    """
    return transforms.Compose([
        transforms.Resize([resize_size, resize_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[target_dataset], std=std[target_dataset]),
    ])

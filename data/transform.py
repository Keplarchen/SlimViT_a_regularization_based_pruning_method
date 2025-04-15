from torchvision import transforms

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

def CIFAR10_transform(target_size: int=224) -> transforms.Compose:
    """

    :param target_size:
    :return:
    """
    return transforms.Compose([
        transforms.Resize([target_size, target_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean['cifar10'], std=std['cifar10']),
    ])

def CIFAR100_transform(target_size: int=224) -> transforms.Compose:
    """

    :param target_size:
    :return:
    """
    return transforms.Compose([
        transforms.Resize([target_size, target_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean['cifar100'], std=std['cifar100']),
    ])

from torchvision import transforms

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

def CIFAR_transform(target_dataset: str,
                    resize_size: int) -> transforms.Compose:
    """
    Constructs a transformation pipeline for preprocessing the CIFAR dataset. The transformations
    include resizing the image, converting it to a tensor, and normalizing it using dataset-specific
    mean and standard deviation values.

    :param target_dataset: The name of the target dataset (e.g., 'cifar10', 'cifar100') for
        selecting the corresponding mean and standard deviation values used in normalization.
    :type target_dataset: str
    :param resize_size: The size to which the images will be resized. The image dimensions will be
        square, resized to [resize_size, resize_size].
    :type resize_size: int
    :return: A ``transforms.Compose`` object containing the series of transformations to apply
        to the dataset, including resizing, tensor conversion, and normalization.
    :rtype: transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize([resize_size, resize_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[target_dataset], std=std[target_dataset]),
    ])

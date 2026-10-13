from torchvision import transforms as T


def cifar_transform(train: bool):
    augmentation = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if train else []
    return T.Compose(augmentation + [T.ToTensor(), T.Normalize((.4914,.4822,.4465), (.247,.243,.261))])

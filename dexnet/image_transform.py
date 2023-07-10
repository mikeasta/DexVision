import torchvision
from torchvision import transforms
from typing import Tuple


def get_image_transform(
        size: Tuple[int, int, int] = (3, 224, 224),
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.NEAREST
) -> torchvision.transforms:
    """
    Returns necessary image transforms before pushing them into model.
    :param size: Sets new image shape.
    :param interpolation: Sets image interpolation mode (nearest, bilinear, bicubic and etc.).
    :return: torchvision transforms object for pushing into dataloader.
    """
    image_transforms = transforms.Compose([
        transforms.Resize(size=size),
        transforms.TrivialAugmentWide(interpolation=interpolation),
        transforms.ToTensor()
    ])

    return image_transforms

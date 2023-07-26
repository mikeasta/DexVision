from torchvision import transforms
from typing import Tuple


class ImageTransforms:
    def __init__(self, image_size: Tuple[int, int]):
        self.image_size = image_size

    @property
    def train_image_transforms(self):
        return transforms.Compose([
            transforms.Resize(size=self.image_size),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor()
        ])

    @property
    def test_image_transforms(self):
        return transforms.Compose([
            transforms.Resize(size=self.image_size),
            transforms.ToTensor()
        ])

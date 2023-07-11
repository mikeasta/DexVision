import torch
import pathlib
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple

from dexnet.classes.DexDataset import DexDataset


def create_dataloaders(target_path: pathlib.Path, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Returns torch.utils.data.DataLoaders objects for loading train and test data consecutive.

    :param target_path: Path to directory with train and test dirs.
    :param batch_size: Batch size for loading into model.
    :return: Train and test dataloaders.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_path = target_path / "train"
    test_path = target_path / "test"

    train_dataset = DexDataset(target_path=train_path, transform=train_transforms)
    test_dataset = DexDataset(target_path=test_path, transform=test_transforms)

    print(f"Train dataset imported and it contains {len(train_dataset)} samples")
    print(f"Test dataset imported and it contains {len(test_dataset)} samples")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    return train_dataloader, test_dataloader

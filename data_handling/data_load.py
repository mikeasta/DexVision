import pathlib
from torch.utils.data import DataLoader
from typing import Tuple

from data_handling.transforms import ImageTransforms
from data_handling.dataset import CustomImageFolder


def create_dataloaders(target_path: pathlib.Path, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Returns torch.utils.data.DataLoaders objects for loading train and test data consecutive.

    :param target_path: Path to directory with train and test dirs.
    :param batch_size: Batch size for loading into model.
    :return: Train and test dataloaders.
    """
    image_transforms = ImageTransforms(image_size=(64, 64))
    train_transforms = image_transforms.train_image_transforms()
    test_transforms = image_transforms.test_image_transforms()

    train_path = target_path / "train"
    test_path = target_path / "test"

    train_dataset = CustomImageFolder(target_path=train_path, transform=train_transforms)
    test_dataset = CustomImageFolder(target_path=test_path, transform=test_transforms)

    print(f"Train dataset imported and it contains {len(train_dataset)} samples")
    print(f"Test dataset imported and it contains {len(test_dataset)} samples")
    print(f"Classes: {train_dataset.class_to_idx}")

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

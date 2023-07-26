import torch
import pathlib
from PIL import Image
from typing import Tuple, Dict, List
from torch.utils.data import Dataset


def find_classes(path: pathlib.Path) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in path.iterdir() if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"No classes in {path} directory.")
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    return classes, class_to_idx


class CustomImageFolder(Dataset):
    def __init__(self, target_path: pathlib.Path, transform) -> None:
        self.paths = list(target_path.glob("*/*.png"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_path)

    def load_image(self, index: int):
        image_path = self.paths[index]
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            image = self.transform(image)
        return image,  class_idx

    def __len__(self) -> int:
        return len(self.paths)

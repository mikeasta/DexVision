import os
from pathlib import Path
from dexnet.utils.seed import set_seed


def split_images_into_train_test(
        train_ratio: float = 0.75,
        shuffle: bool = False
) -> None:
    """
    Takes images from data/images directory and divides them into data/train and data/test directories.
    Images might be ordered randomly by using `shuffle` parameter.

    :param train_ratio: Percent of train images
    :param shuffle: Randomized image order necessity
    :return:
    """
    data_path = Path('../../data/pokemons')
    images_path = data_path / "images"
    train_path = data_path / "train"
    test_path = data_path / "test"

    class_names = [class_dir.name for class_dir in images_path.iterdir() if class_dir.is_dir()]
    print(f"Classes: {class_names}")

    for class_name in class_names:
        pass
        # 1. Save images list

        # 2. Make random sample for train and then test images

        # 3. Save images in consecutive folders


if __name__ == "__main__":
    split_images_into_train_test(shuffle=True)

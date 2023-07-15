import os
import pathlib


def order_rename(images_path: pathlib.Path, start_index: int = 1) -> None:
    """
    Rename images to ordered in all folders of `images_path`.

    :param images_path: All images in this path will be renamed
    :param start_index: Start value
    """
    image_names = [image.name for image in images_path.glob("*.png")]
    for loop_index, image in enumerate(image_names):
        os.rename(images_path / image, images_path / f"{loop_index + start_index}.png")


def rename_all() -> None:
    """
    [PATHS MUTABLE]:
    Renames all images in test and train directories for all classes
    """
    data_path = pathlib.Path("../data/pokemons")
    train_path = data_path / "train"
    test_path = data_path / "test"

    class_names = [class_dir.name for class_dir in train_path.iterdir() if class_dir.is_dir()]
    for class_name in class_names:
        order_rename(train_path / class_name)
        order_rename(test_path / class_name)


if __name__ == "__main__":
    order_rename(images_path=pathlib.Path("../data/input/"))
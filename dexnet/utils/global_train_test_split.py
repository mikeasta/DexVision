import os
import random
import shutil
import pathlib
from dexnet.utils import seed


def split_images_into_train_test(
        train_ratio: float = 0.75,
        shuffle: bool = False
) -> None:
    """
    Takes images from `**/images` directory and divides them into `**/train` and `**/test` directories.
    Images might be ordered randomly by using `shuffle` parameter.

    :param train_ratio: Percent of train images.
    :param shuffle: Should we shuffle images while splitting?
    :return:
    """
    data_path = pathlib.Path('../../data/pokemons')
    images_path = data_path / "images"
    train_path = data_path / "train"
    test_path = data_path / "test"

    class_names = [class_dir.name for class_dir in images_path.iterdir() if class_dir.is_dir()]
    print(f"Classes: {class_names}")

    for class_name in class_names:
        class_path = images_path / class_name
        images = [image.name for image in list(class_path.glob("*.png"))]

        if shuffle:
            seed.set_seed()
            train_images_sample = random.sample(population=images, k=int(len(images) * train_ratio))
        else:
            train_images_sample = images[:int(len(images) * train_ratio)-1]

        test_images_sample = [image for image in images if image not in train_images_sample]

        # 3. Save images in consecutive folders
        copy_path = images_path / class_name
        train_save_path = train_path / class_name
        test_save_path = test_path / class_name

        if not os.path.isdir(train_save_path):
            os.mkdir(path=train_save_path)

        if not os.path.isdir(test_save_path):
            os.mkdir(path=test_save_path)

        for image in train_images_sample:
            shutil.copy(copy_path / image, train_save_path)

        for image in test_images_sample:
            shutil.copy(copy_path / image, test_save_path)

        # Logging
        train_files = [image.name for image in train_save_path.iterdir()]
        test_files = [image.name for image in test_save_path.iterdir()]
        print(f"For class {class_name} there are: {len(train_files)} train images and {len(test_files)} test images.")


if __name__ == "__main__":
    split_images_into_train_test(shuffle=True)

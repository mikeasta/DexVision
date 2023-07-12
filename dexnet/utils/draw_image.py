import os
import pathlib
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def draw_random_images(amount: int = 1, directory: str = None) -> None:
    """
    Draws some amount of PNG images from `directory` dir in random order.

    :param amount: Amount of pictures to draw.
    :param directory: Directory from which we should pick images
    """
    path = pathlib.Path(directory)
    images_paths = [image_path for image_path in path.glob("*.png")]

    if not images_paths:
        raise FileNotFoundError(f"Directory {path} is empty (req. PNG images) or doesn't exist.")

    plt.figure(figsize=(10, 6))

    transform = transforms.Compose([
        transforms.Resize((64, 64))
    ])

    for i in range(amount):
        image_path = random.choice(images_paths)
        image = transform(Image.open(image_path))
        plt.subplot(1, amount, i+1)
        plt.imshow(image)
        plt.title(image_path.parent.name)
        plt.axis(False)
        plt.savefig("../plots/random_images")


def draw_image(image: Image, save_path: str, title: str = None) -> None:
    """
    Draws special image, opened with PIL.Image, and saves it.

    :param image: PIL.Image representation.
    :param save_path: Save image directory.
    :param title: Plot title.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis(False)
    plt.savefig(save_path)


if __name__ == "__main__":
    draw_random_images(directory="../../data/pokemons/train/Squirtle/", amount=5)

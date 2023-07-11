import torch
import timeit
import pathlib
import matplotlib.pyplot as plt
import torchvision
import dexnet.dexio as dex_io
import dexnet.utils.device_check as dex_device_check
from torchvision import transforms
from dexnet.classes.DexDataset import DexDataset
from dexnet.classes.PokemonClassifierModel import PokemonClassifierModel


def test_images(images_amount: int = 1, pick_train: bool = True) -> None:
    """
    Performs model for certain amount of random images and draws a plot

    :param images_amount: Amount of evaluated images
    :param pick_train:
    :return:
    """

def test_evaluate() -> None:
    device = dex_device_check.get_best_device()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = DexDataset(pathlib.Path("../data/pokemons/train"), transform)
    model = PokemonClassifierModel().to(device)

    image, label = train_dataset[0]

    # TODO: make function for single image test (or other amount)
    model.eval()
    with torch.inference_mode():
        result = model(image.unsqueeze(dim=0))

    # TODO: make plot show
    print(result)
    plt.imshow(image.permute(1, 2, 0))
    plt.axis(False)
    plt.title(f"True: {label}")


if __name__ == "__main__":
    # Evaluate model
    evaluate()

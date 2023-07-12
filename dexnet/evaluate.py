import os
import torch
import pathlib
from torchvision import transforms
from PIL import Image

from dexnet.classes.PokemonClassifierModel import PokemonClassifierModel
from dexnet import dexio
from dexnet.utils.draw_image import draw_image


def evaluate_on_images(input_dir: str, output_dir: str):
    """
    Evaluates model on one single image.

    :param input_dir: Directory to take image.
    :param output_dir: Directory to save image.
    :return:
    """

    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(f"There is no {input_path} directory.")

    if not output_path.is_dir():
        os.mkdir(output_dir)

    images = [image for image in input_path.glob("*.png")]

    if not images:
        raise FileNotFoundError(f"There is no PNG images in {input_path} directory.")

    model = PokemonClassifierModel()
    model.load_state_dict(dexio.load_weights())

    image_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        transformed_image = image_transform(image).unsqueeze(dim=0)
        y_log = model(transformed_image)
        prob = torch.softmax(y_log, dim=1).max()
        res = y_log.argmax(dim=1)
        classes = ["Bulbasaur", "Charmander", "Pikachu", "Squirtle"]

        image_name = f"evaluate_{image_path.name[:-4]}"
        draw_image(image=image,
                   title=f"Predicted {classes[res]} with {prob*100:.2f}% likelihood.",
                   save_path=str(output_path / image_name))


if __name__ == "__main__":
    evaluate_on_images(input_dir="../data/input",
                       output_dir="../data/output")

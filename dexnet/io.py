import os
import pathlib
import torch

DEFAULT_MODEL_SAVE_PATH = pathlib.Path("models/")


def save_weights(model: torch.nn.Module, path: pathlib.Path = DEFAULT_MODEL_SAVE_PATH) -> None:
    """
    Saves model weights in folder to be able to run this model later.
    :param model: NN model
    :param path: Path to save
    """
    if not path.is_dir():
        os.mkdir(path)

    torch.save(model.state_dict(), path)


def load_weights(path: pathlib.Path = DEFAULT_MODEL_SAVE_PATH, weights_save_name: str = None):
    """
    Loads model weights
    :param path: Directory with model's weights
    :param weights_save_name: Special model name. If None, loading last model
    :return: NN weights in dictionary
    """
    if not path.is_dir():
        raise FileNotFoundError(f"There is no directory '{path}'")

    if weights_save_name:
        weights_path = path / weights_save_name

        if not weights_path.is_file():
            raise FileNotFoundError(f"There is no file '{weights_path}'")

        return torch.load(weights_path)
    else:
        weights_saves = [save.name for save in path.iterdir() if ".pth" in save.name or "pt" in save.name]
        weights_saves.sort()
        last_weight = weights_saves[-1]
        return torch.load(path/last_weight)

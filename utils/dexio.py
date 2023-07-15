import os
import pathlib
import torch
from datetime import datetime
DEFAULT_MODEL_SAVE_PATH = pathlib.Path("../models/")


def save_weights(model: torch.nn.Module, path: pathlib.Path = DEFAULT_MODEL_SAVE_PATH) -> None:
    """
    Saves model weights in folder to be able to run this model later.
    :param model: NN model
    :param path: Path to save
    """
    if not path.is_dir():
        os.mkdir(path)

    now = datetime.now()
    save_name = now.strftime("%m_%d_%Y__%H_%M_%S") + ".pth"
    torch.save(model.state_dict(), path / save_name)


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

        print(f"Loads {weights_path} model weights")
        return torch.load(weights_path)
    else:
        weights_saves = [save.name for save in path.iterdir() if ".pth" in save.name or ".pt" in save.name]
        weights_saves.sort()
        last_weight = weights_saves[-1]
        print(f"Loads {path/last_weight} model weights")
        return torch.load(path/last_weight)

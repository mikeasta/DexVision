import pathlib
import torch
from dexnet.data_load import create_dataloaders


def train(
        data_directory: str,
        model_save_directory: str = None,
        model: torch.nn.Module = None,
        epochs: int = 50
) -> None:
    """
    Train baseline for
    :param data_directory: Directory, from which images would be loaded
    :param model_save_directory: Directory to save current model state
    :param model: torch.nn.Module model
    :param epochs: Amount of epochs
    :return:
    """
    # 1. Load data
    data_path = pathlib.Path(data_directory)
    train_dataloader, test_dataloader = create_dataloaders(data_path)


if __name__ == "__main__":
    train("../data/pokemons/")

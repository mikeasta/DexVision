import torch
import random


def set_seed(seed: int = 42) -> None:
    """
    Sets special random seed for pseudorandom parameters and values generation in Torch and in general.

    Required to be called before every pseudorandom generation performance.

    :param seed: Random seed value.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

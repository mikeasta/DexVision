import torch


def set_device() -> str:
    """
    Defines best usable computational device (CPU, GPU or etc.).
    :return: Device label string.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

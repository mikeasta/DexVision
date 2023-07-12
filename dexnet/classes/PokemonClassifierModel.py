import torch
from torch import nn


class PokemonClassifierModel(nn.Module):
    """
        For this project was created a model, which architecture was inspired by VGG16 architecture.
        This model architecture can be labeled as clone of AlexNet.
        But channels amount and convolutional layer sizes were increased.
    """
    def __init__(self):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)

        return x

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
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=128,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.classifier(x)

        return x

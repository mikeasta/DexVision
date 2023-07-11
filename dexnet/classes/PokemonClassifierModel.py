import torch
from torch import nn


class PokemonClassifierModel(nn.Module):
    """
        For this project was created a model, which architecture was inspired by VGG16 architecture.
        This model architecture can be labeled as VGG13. Model summary:

        - Block 1:
            - Input: 224*224*3*1
            - Conv2d: 224*224*3*16
            - ReLU
            - Conv2d: 224*224*3*16
            - ReLU
            - MaxPool2d: kernel=2, stride=2
        - Block 2
            - Conv2d: 112*112*3*20
            - ReLU
            - Conv2d: 112*112*3*20
            - ReLU:
            - MaxPool2d: kernel=2, stride=2
        - Block 3:
            - Conv2d: 56*56*3*24
            - ReLU
            - Conv2d: 56*56*3*24
            - ReLU
            - MaxPool2d: kernel=2, stride=2
        - Block 4:
            - Conv2d: 28*28*3*28
            - ReLU
            - Conv2d: 28*28*3*28
            - ReLU
            - MaxPool2d
        - Block 5:
            - Conv2d: 14*14*3*32
            - ReLU
            - Conv2d: 14*14*3*32
            - ReLU
            - MaxPool2d
            - Conv2d: 7*7*3*36
        - Classifier
            - Flatten
            - Linear: 1764 -> 64
            - Linear: 64 -> 64
            - Linear 64 -> 4
    """
    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
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

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=20,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=20,
                out_channels=20,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=24,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=24,
                out_channels=24,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=24,
                out_channels=28,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=28,
                out_channels=28,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=28,
                out_channels=32,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                padding=1,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                in_channels=32,
                out_channels=36,
                padding=1,
                kernel_size=3
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1764,64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.classifier(x)

        return x

from torch import nn
import torch


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()

        self.convlayers = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2), # 1 X 32 X 32 X 32 images
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p = 0.4),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2), # 1 X 64 X 16 X 16 images
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p = 0.4)
        )
        self.linlayers = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(128, 10),
        )

        
    def forward(self, x):
        covn_output = self.convlayers(x)
        linear = torch.flatten(covn_output, 1)
        out = self.linlayers(linear)
        return nn.functional.log_softmax(out, dim = 1)
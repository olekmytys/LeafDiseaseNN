# model.py
import torch.nn as nn

class LeafCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128 -> 64x64 (400->200)

            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 32x32 (200->100)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28800, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.classifier(x)
        return x
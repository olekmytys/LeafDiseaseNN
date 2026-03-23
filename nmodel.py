import torch.nn as nn

class LeafCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),


            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),


            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.classifier(x)
        return x
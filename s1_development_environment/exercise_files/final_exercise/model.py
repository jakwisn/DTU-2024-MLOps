from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 28x28
        self.pool1 = nn.MaxPool2d(2, 2) # 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 14x14
        self.pool2 = nn.MaxPool2d(2, 2) # 7x7

        self.fc = nn.Linear(64*7*7, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.pool1(x))
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.shape[0], -1)
        x = F.log_softmax(self.fc(x), dim=1)

        return x

import torch
import torch.nn as nn


class ANNConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        spk1 = torch.relu(self.pool1(self.conv1(x)))
        spk2 = torch.relu(self.pool2(self.conv2(spk1)))
        flt = spk2.view(-1, 16 * 4 * 4)  # B, 16*5*5
        spk3 = self.fc1(flt)

        return spk3

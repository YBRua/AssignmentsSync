import torch
import torch.nn as nn


class ANNMLP(nn.Module):
    def __init__(self, input_dims: int, hid_dims: int, output_dims: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, hid_dims)
        self.fc2 = nn.Linear(hid_dims, output_dims)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

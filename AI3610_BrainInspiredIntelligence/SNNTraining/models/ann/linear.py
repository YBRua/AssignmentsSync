import torch
import torch.nn as nn


class ANNLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        x = self.fc1(x)

        return x

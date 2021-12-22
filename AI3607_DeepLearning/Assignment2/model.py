import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LeNet(nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2D(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2D(kernel_size=2)
        self.conv2 = nn.Conv2D(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2D(kernel_size=2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

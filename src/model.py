from torch.nn import Module
from torch import nn


class SimpleConvNet(Module):
    def __init__(self, classes):
        super(SimpleConvNet, self).__init__()
        # input shape: (batch_size, 3, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*16*16, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

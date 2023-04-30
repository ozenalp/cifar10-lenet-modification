import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet_0(nn.Module):
    def __init__(self):
        super(LeNet_0, self).__init__()

        # Layer 1: Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2: Convolutional layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3: Fully connected layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        # Layer 4: Fully connected layer
        self.fc2 = nn.Linear(120, 84)

        # Layer 5: Output layer
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = [batch size, 3, 32, 32]
        # Layer 1: Convolutional layer
        x = self.conv1(x)
        # x = [batch size, 6, 28, 28]
        x = F.relu(x)
        x = self.pool1(x)
        # x = [batch size, 6, 14, 14]

        # Layer 2: Convolutional layer
        x = self.conv2(x)
        # x = [batch size, 16, 10, 10]
        x = F.relu(x)
        x = self.pool2(x)
        # x = [batch size, 16, 5, 5]

        # Flatten the output from layer 2
        x = x.view(-1, 16 * 5 * 5)
        h = x

        # Layer 3: Fully connected layer
        x = self.fc1(x)
        # x = [batch size, 120]
        x = F.relu(x)

        # Layer 4: Fully connected layer
        x = self.fc2(x)
        # x = [batch size, 84]
        x = F.relu(x)

        # Layer 5: Output layer
        x = self.fc3(x)
        # x = [batch size, output_dim]

        return x, h
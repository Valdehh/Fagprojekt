import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class encoder(nn.Module):
    def __init__(self) -> None:
        super(encoder).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(32 * 68 * 68, 2 * 200)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32 * 68 * 68)
        x = self.fully_connected(x)
        return x


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.input = nn.Linear(2 * 200, 32 * 68 * 68)
        self.conv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=5, stride=1, padding=2),
        self.conv2 = nn.ConvTranspose2d(
            16, 3, kernel_size=5, stride=1, padding=2)
        self.fully_connected = nn.Linear(3 * 68 * 68, 3 * 68 * 68 * 256)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32, 68, 68)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 3 * 32 * 32)
        x = self.fully_connected(x)
        x = x.view(-1, 3 * 68 * 68, 256)
        x = self.softmax(x)
        return x

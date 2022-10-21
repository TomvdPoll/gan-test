import torch
from torch import nn, optim
from torch.nn import functional as F


class SimpleGenerator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(SimpleGenerator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=5e-5)
        self.loss = nn.BCELoss()

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Conv1dGenerator(nn.Module):
    def __init__(self, noise_size, channels_img, features_g):
        super(Conv1dGenerator, self).__init__()
        self.gen = nn.Sequential(
            # input: N x noise_size x 1
            self._block(noise_size, features_g * 64, kernel_size=4, stride=1, padding=0, output_padding=0),  # 4
            self._block(features_g * 64, features_g * 32, kernel_size=4, stride=2, padding=0, output_padding=0),  # 10
            self._block(features_g * 32, features_g * 16, kernel_size=4, stride=2, padding=0, output_padding=0),  # 22
            self._block(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=0, output_padding=1),  # 46+1
            self._block(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=0, output_padding=0),  # 96
            self._block(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=0, output_padding=0),  # 194
            self._block(features_g * 2, features_g, kernel_size=4, stride=2, padding=0, output_padding=1),  # 390+1
            nn.ConvTranspose1d(features_g, channels_img, kernel_size=4, stride=2, padding=0, output_padding=0, bias=False),  # 784
            nn.Tanh()

        )
        self.optimizer = optim.Adam(self.parameters(), lr=5e-5, betas=(0.5, 0.999))
        self.loss = nn.BCELoss()

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding, output_padding):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)

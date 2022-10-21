import torch
from torchgan.layers import MinibatchDiscrimination1d
from torch import nn, optim
from torch.nn import functional as F


class SimpleDiscriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(SimpleDiscriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=5e-5)
        self.loss = nn.BCELoss()

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        # x = F.leaky_relu(self.minibatchdisc(x), 0.2)
        return torch.sigmoid(self.fc4(x))


class Conv1dDiscriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Conv1dDiscriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 784
            nn.Conv1d(channels_img, features_d, kernel_size=4, stride=2, padding=0, bias=False),  # 391
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, kernel_size=4, stride=2, padding=0),  # 194
            self._block(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=0),  # 96
            self._block(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=0),  # 47
            self._block(features_d * 8, features_d * 16, kernel_size=4, stride=2, padding=0),  # 22
            self._block(features_d * 16, features_d * 32, kernel_size=4, stride=2, padding=0),  # 10
            self._block(features_d * 32, features_d * 64, kernel_size=4, stride=2, padding=0),  # 4
            nn.Conv1d(features_d * 64, 1, kernel_size=4, stride=1, padding=0, bias=False),  # 1
            nn.Sigmoid()

        )
        self.optimizer = optim.Adam(self.parameters(), lr=5e-5, betas=(0.5, 0.999))
        self.loss = nn.BCELoss()

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):

        return self.disc(x)


if __name__ == "__main__":
    disc = Conv1dDiscriminator(channels_img=1, features_d=64)
    a=2

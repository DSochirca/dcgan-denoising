from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, kernel_size=3, num_layers=2):
        super(ResidualBlock, self).__init__()
        layers = []

        for i in range(num_layers):
            layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(num_channels))
            layers.append(nn.ReLU())

        self.residual_block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.residual_block(x)
        y = y + x   # Residual connection
        return y

class ResidualLayer(nn.Module):
    def __init__(self, num_channels, kernel_size=3, downsampling_factor=1):
        super(ResidualLayer, self).__init__()
        self.downsampling_factor = downsampling_factor

        self.residual_block = ResidualBlock(num_channels, kernel_size, num_layers=2)

        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        self.batch_norm = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()

        self.conv_transpose = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        y = self.residual_block(x)

        if self.downsampling_factor > 1:
            y = self.upscale(y)
            y = self.batch_norm(y)
            y = self.relu(y)

            y = self.conv_transpose(y)

        return y

class Generator(nn.Module):
    def __init__(self, in_channels, residual_channels=None, kernel_size=3, downsampling_factor=1):
        super(Generator, self).__init__()

        if residual_channels is None:  residual_channels = [64, 64, 64, 64]

        self.downsampling_factor = downsampling_factor

        self.conv1 = nn.Conv2d(in_channels, residual_channels[0], kernel_size=kernel_size, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        K = downsampling_factor
        assert K == 1 or K == 2 or K == 4 or K == 8

        residual_layers = []
        for i in range(len(residual_channels)):
            residual_layers.append(ResidualLayer(residual_channels[i], kernel_size, K))
            K = K // 2  # Spatial upscale

        self.residual_layers = nn.Sequential(*residual_layers)

        self.conv2 = nn.Conv2d(residual_channels[-1], residual_channels[-1], kernel_size=kernel_size, stride=1, padding=1)

        self.conv3 = nn.Conv2d(residual_channels[-1], residual_channels[-1], kernel_size=kernel_size, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(residual_channels[-1], residual_channels[-1], kernel_size=1, stride=1, padding=1)  # last 2 conv use kernel_size=1
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(residual_channels[-1], in_channels, kernel_size=1, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)

        residual = y
        y = self.residual_layers(y)

        y = self.conv2(y)  # before the residual connection below, because it needs the same number of channels

        if self.downsampling_factor == 1:
            y = y + residual

        y = self.relu3(self.conv3(y))
        y = self.relu4(self.conv4(y))
        y = self.conv5(y)
        y = self.sigmoid(y)

        return y

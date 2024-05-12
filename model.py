import torch
import torch.nn as nn
import numpy as np

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

class Discriminator(nn.Module):
    def __init__(self, input_size, in_channels, hidden_channels=None, kernel_size=3):
        super(Discriminator, self).__init__()
        output_size = input_size

        if hidden_channels is None:  hidden_channels = [64, 64, 128, 256, 512]  # an extra 64, as in the original model

        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], kernel_size=kernel_size, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2)

        layers = []
        for i in range(1, len(hidden_channels)):
            layers.append(nn.Conv2d(hidden_channels[i-1], hidden_channels[i], kernel_size=kernel_size, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(hidden_channels[i]))
            layers.append(nn.LeakyReLU(0.2))

        self.hidden_layers = nn.Sequential(*layers)

        self.conv2 = nn.Conv2d(hidden_channels[-1], hidden_channels[-1], kernel_size=kernel_size, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(hidden_channels[-1])
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(hidden_channels[-1], hidden_channels[-1], kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(hidden_channels[-1])
        self.lrelu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(hidden_channels[-1], 1, kernel_size=1, stride=1, padding=0)

        output_size = output_size // 4**(len(hidden_channels) - 1)  # (#hidden_channels - 1) layers. Power of 4 because we have 2d images (2*2) - output gets halved in each dimension
        output_size = output_size * 1  # 1 is the #channels in last conv layer
        # print('Output size:', output_size)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(output_size, 1024)
        self.lrelu4 = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(1024, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.lrelu1(y)

        y = self.hidden_layers(y)

        y = self.lrelu2(self.batch_norm2(self.conv2(y)))
        y = self.lrelu3(self.batch_norm3(self.conv3(y)))

        y = self.conv4(y)

        y = self.flatten(y)
        y = self.dense1(y)
        y = self.lrelu4(y)
        y = self.dense2(y)

        return y

if __name__ == '__main__':
    # Dummy input (1 image w/ 3 channels, 256x256)
    np.random.seed(42)
    image = np.random.rand(1, 3, 256, 256).astype(np.float32)
    image = torch.from_numpy(image)

    # Create the model
    generator = Generator(in_channels=3)
    discriminator = Discriminator(input_size=256*256, in_channels=3)

    # Forward pass
    output = generator(image)
    print('Generator output shape:', output.shape)

    output = discriminator(image)
    print('Discriminator output shape:', output.shape)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

    # Training loop
    generator.train()
    discriminator.train()
    gen_losses = []
    disc_losses = []

    for epoch in range(10):
        discriminator_optimizer.zero_grad()
        generator_optimizer.zero_grad()

        fake_image = generator(image)
        real_disc = discriminator(image)
        fake_disc = discriminator(fake_image)

        # Discriminator
        real_loss = criterion(real_disc, torch.ones_like(real_disc))
        fake_loss = criterion(fake_disc, torch.zeros_like(fake_disc))

        disc_loss = real_loss + fake_loss
        disc_loss.backward()

        discriminator_optimizer.step()

        # Generator
        fake_image = generator(image)
        fake_disc = discriminator(fake_image)
        gen_loss = criterion(fake_disc, torch.ones_like(fake_disc))
        gen_loss.backward()

        generator_optimizer.step()

        gen_losses.append(gen_loss.item())
        disc_losses.append(disc_loss.item())

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{10}], Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}')

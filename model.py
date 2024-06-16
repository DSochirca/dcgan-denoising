import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
class FactorizedSelfAttention(nn.Module):
    def __init__(self, in_channels, embed_channels=None):
        super(FactorizedSelfAttention, self).__init__()
        if embed_channels is None:
            embed_channels = in_channels // 8

        self.C_embed = embed_channels

        self.query_conv = nn.Conv2d(in_channels, embed_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, embed_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        C_embed = self.C_embed

        # Project to query, key, value
        # -> B, embed_channels, W, H
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        # Reshape for row and column attention
        Q_row = Q.permute(0, 2, 1, 3).contiguous().view(B * W, H, C_embed)  # (B, W, H, C_embed) -> (B*W, H, C_embed)
        K_row = K.permute(0, 2, 1, 3).contiguous().view(B * W, H, C_embed)
        V_row = V.permute(0, 2, 1, 3).contiguous().view(B * W, H, C)

        # Row-wise attention
        # A_row should be (B*W, H, H)
        A_row = self.softmax(torch.bmm(Q_row, K_row.transpose(1, 2))) # (B*W, H, C_embed) x (B*W, C_embed, H) -> (B*W, H, H)
        O_row = torch.bmm(A_row, V_row)  # (B*W, H, H) x (B*W, H, C) -> (B*W, H, C)
        O_row = O_row.view(B, W, H, C).permute(0, 3, 1, 2)  # -> (B, C, W, H)

        # Reshape for column attention
        Q_col = O_row.permute(0, 3, 2, 1).contiguous().view(B * H, W, C) # (B, C, W, H) -> (B, H, W, C) -> (B*H, W, C)
        K_col = O_row.permute(0, 3, 2, 1).contiguous().view(B * H, W, C)
        V_col = O_row.permute(0, 3, 2, 1).contiguous().view(B * H, W, C)

        # Column-wise attention
        # A_col should be (B*H, W, W)
        A_col = self.softmax(torch.bmm(Q_col, K_col.transpose(1, 2)))  # (B*H, W, C) x (B*H, C, W) -> (B*H, W, W)
        O_col = torch.bmm(A_col, V_col)  # (B*H, W, W) x (B*H, W, C) -> (B*H, W, C)
        O_col = O_col.view(B, H, W, C).permute(0, 3, 2, 1)  # -> (B, C, W, H)

        # Combine with residual connection
        y = self.gamma * O_col + x
        return y

class ResidualBlock(nn.Module):
    def __init__(self, num_channels, kernel_size=3, num_layers=2):
        super(ResidualBlock, self).__init__()
        layers = []

        for i in range(num_layers):
            layers.append(spectral_norm(nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=1)))
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
    def __init__(self, in_channels, residual_channels=None, kernel_size=3, downsampling_factor=1, uses_self_attention=False):
        super(Generator, self).__init__()

        if residual_channels is None:  residual_channels = [64, 64, 64, 64]

        self.downsampling_factor = downsampling_factor
        self.uses_self_attention = uses_self_attention

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, residual_channels[0], kernel_size=kernel_size, stride=1, padding=1))
        self.relu1 = nn.ReLU()

        K = downsampling_factor
        assert K == 1 or K == 2 or K == 4 or K == 8

        # Self-attention before 1st residual layer
        if uses_self_attention:
            self.attn1 = FactorizedSelfAttention(residual_channels[0], embed_channels=8)

        residual_layers = []
        for i in range(len(residual_channels)):
            residual_layers.append(ResidualLayer(residual_channels[i], kernel_size, K))
            K = K // 2  # Spatial upscale

        self.residual_layers = nn.Sequential(*residual_layers)

        # Self-attention after last residual layer
        if uses_self_attention:
            self.attn2 = FactorizedSelfAttention(residual_channels[-1], embed_channels=8)

        self.conv2 = spectral_norm(nn.Conv2d(residual_channels[-1], residual_channels[-1], kernel_size=kernel_size, stride=1, padding=1))

        self.conv3 = spectral_norm(nn.Conv2d(residual_channels[-1], residual_channels[-1], kernel_size=kernel_size, stride=1, padding=1))
        self.relu3 = nn.ReLU()
        self.conv4 = spectral_norm(nn.Conv2d(residual_channels[-1], residual_channels[-1], kernel_size=1, stride=1, padding=0))  # last 2 conv use kernel_size=1
        self.relu4 = nn.ReLU()
        self.conv5 = spectral_norm(nn.Conv2d(residual_channels[-1], in_channels, kernel_size=1, stride=1, padding=0))  # padding=0 if kernel_size=1

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)

        if self.uses_self_attention:
            y = self.attn1(y)

        residual = y
        y = self.residual_layers(y)

        if self.uses_self_attention:
            y = self.attn2(y)

        y = self.conv2(y)  # before the residual connection below, because it needs the same number of channels

        if self.downsampling_factor == 1:
            y = y + residual

        y = self.relu3(self.conv3(y))
        y = self.relu4(self.conv4(y))
        y = self.conv5(y)
        y = self.sigmoid(y)

        return y

class Discriminator(nn.Module):
    def __init__(self, input_size, in_channels, hidden_channels=None, kernel_size=3, uses_self_attention=False):
        super(Discriminator, self).__init__()
        output_size = input_size

        if hidden_channels is None:  hidden_channels = [64, 64, 128, 256, 512]  # an extra 64, as in the original model
        self.uses_self_attention = uses_self_attention

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, hidden_channels[0], kernel_size=kernel_size, stride=1, padding=1))
        self.lrelu1 = nn.LeakyReLU(0.2)

        # Self-attention before 1st hidden layer
        if uses_self_attention:
            self.attn1 = FactorizedSelfAttention(hidden_channels[0], embed_channels=8)

        layers = []
        for i in range(1, len(hidden_channels)):
            layers.append(spectral_norm(nn.Conv2d(hidden_channels[i-1], hidden_channels[i], kernel_size=kernel_size, stride=2, padding=1)))
            layers.append(nn.BatchNorm2d(hidden_channels[i]))
            layers.append(nn.LeakyReLU(0.2))

        self.hidden_layers = nn.Sequential(*layers)

        # Self-attention after last hidden layer
        if uses_self_attention:
            self.attn2 = FactorizedSelfAttention(hidden_channels[-1], embed_channels=8)

        self.conv2 = spectral_norm(nn.Conv2d(hidden_channels[-1], hidden_channels[-1], kernel_size=kernel_size, stride=1, padding=1))
        self.batch_norm2 = nn.BatchNorm2d(hidden_channels[-1])
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.conv3 = spectral_norm(nn.Conv2d(hidden_channels[-1], hidden_channels[-1], kernel_size=1, stride=1, padding=0))
        self.batch_norm3 = nn.BatchNorm2d(hidden_channels[-1])
        self.lrelu3 = nn.LeakyReLU(0.2)

        self.conv4 = spectral_norm(nn.Conv2d(hidden_channels[-1], 1, kernel_size=1, stride=1, padding=0))

        output_size = output_size // 4**(len(hidden_channels) - 1)  # (#hidden_channels - 1) layers. Power of 4 because we have 2d images (2*2) - output gets halved in each dimension
        output_size = output_size * 1  # 1 is the #channels in last conv layer
        # print('Output size:', output_size)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(output_size, 1024)
        self.lrelu4 = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.lrelu1(y)

        if self.uses_self_attention:
            y = self.attn1(y)

        y = self.hidden_layers(y)

        if self.uses_self_attention:
            y = self.attn2(y)

        y = self.lrelu2(self.batch_norm2(self.conv2(y)))
        y = self.lrelu3(self.batch_norm3(self.conv3(y)))

        y = self.conv4(y)

        y = self.flatten(y)
        y = self.dense1(y)
        y = self.lrelu4(y)
        y = self.dense2(y)
        y = self.sigmoid(y)

        return y
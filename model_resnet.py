# Self-attention GAN implementation by Christian Cosgrove
# Based on the paper by Zhang et al.
# https://arxiv.org/abs/1805.08318

# ResNet generator and discriminator
import torch
from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm
from conditional_batch_norm import ConditionalBatchNorm2d
from self_attention import SelfAttention
from self_attention import SelfAttentionPost
import numpy as np


num_classes = 10
channels = 3

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.conv1 = SpectralNorm(self.conv1)
        self.conv2 = SpectralNorm(self.conv2)

        self.bn1 = ConditionalBatchNorm2d(in_channels, num_classes)

        self.upsample1 = nn.Upsample(scale_factor=2)

        self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)

        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x, label):
        m = x
        m = self.bn1(m, label)
        m = nn.ReLU()(m)
        m = self.upsample1(m)
        m = self.conv1(m)
        m = self.bn2(m, label)
        m = nn.ReLU()(m)
        m = self.conv2(m)

        m = m + self.bypass(x)

        return m


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

GEN_SIZE=128
DISC_SIZE=128

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.block1 = ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2)
        self.block2 = ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2)
        self.block3 = ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2)
        self.bn1 = ConditionalBatchNorm2d(GEN_SIZE, num_classes)

    def forward(self, z, label):

        m = self.dense(z).view(-1, GEN_SIZE, 4, 4)
        m = self.block1(m, label)
        m = self.block2(m, label)
        m = self.block3(m, label)
        m = self.bn1(m, label)
        m = nn.ReLU()(m)
        m = self.final(m)
        m = nn.Tanh()(m)

        return m

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.first = FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2)
        self.block1 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2)
        self.block2 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE)

        self.attention_size = 16
        self.att = SelfAttention(128, self.attention_size)
        self.att_post = SelfAttentionPost(128, self.attention_size)

        self.block3 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE)
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)
        self.embed = SpectralNorm(nn.Linear(num_classes, DISC_SIZE))

    def forward(self, x, label):

        m = x
        m = self.first(m)
        m = self.block1(m)
        m = self.block2(m)

        self.attention_output = self.att(m)

        m = self.att_post(m, self.attention_output)

        m = self.block3(m)
        m = nn.ReLU()(m)
        m = self.pool(m)
        m = m.view(-1,DISC_SIZE)
        proj = torch.bmm(m.view(-1, 1, DISC_SIZE), self.embed(label).view(-1, DISC_SIZE, 1))
        return self.fc(m) + proj
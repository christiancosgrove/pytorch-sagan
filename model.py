# Self-attention GAN implementation by Christian Cosgrove
# Based on the paper by Zhang et al.
# https://arxiv.org/abs/1805.08318

# DCGAN-like generator and discriminator
from torch import nn
import torch.nn.functional as F
import torch
from spectral_normalization import SpectralNorm
from conditional_batch_norm import ConditionalBatchNorm2d

channels = 3
leak = 0.1
num_classes = 10
w_g=4

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.conv1 = SpectralNorm(nn.ConvTranspose2d(z_dim, 512, 4, stride=1))
        self.bn1 = ConditionalBatchNorm2d(512, num_classes)
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1)))
        self.bn2 = ConditionalBatchNorm2d(256, num_classes)
        self.conv3 = SpectralNorm(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)))
        self.bn3 = ConditionalBatchNorm2d(128, num_classes)
        self.conv4 = SpectralNorm(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)))
        self.bn4 = ConditionalBatchNorm2d(64, num_classes)
        self.conv5 = SpectralNorm(nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1,1)))

    def forward(self, z, label):

        x = z.view(-1, self.z_dim, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x, label)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x, label)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = self.bn3(x, label)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = self.bn4(x, label)
        x = nn.ReLU()(x)
        x = self.conv5(x)
        x = nn.Tanh()(x)

        return x


attention_size = 8

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))


        #attention layers
        self.f = SpectralNorm(nn.Conv2d(128, attention_size, 1, stride=1))
        self.g = SpectralNorm(nn.Conv2d(128, attention_size, 1, stride=1))
        self.h = SpectralNorm(nn.Conv2d(128, attention_size, 1, stride=1))
        self.i = SpectralNorm(nn.Conv2d(attention_size, 128, 1, stride=1))

        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))

        self.embed = SpectralNorm(nn.Linear(num_classes, w_g * w_g * 512))

        self.gamma = nn.Parameter(torch.tensor(0.))

        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x, c):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))

        f = self.f(m)
        g = self.g(m)
        att = torch.bmm(torch.transpose(f.view(-1, 8 * 8, attention_size), 1, 2), g.view(-1, 8 * 8, attention_size))

        h = self.gamma * self.h(m)
        h = att.view(-1, 8, 8, 1).expand_as(h) + h
        m = self.i(h) + m

        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))
        m = m.view(-1,w_g * w_g * 512)


        return self.fc(m) + torch.bmm(m.view(-1, 1, w_g * w_g * 512), self.embed(c).view(-1, w_g * w_g * 512, 1))


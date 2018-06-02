# Implementation of self-attention layer by Christian Cosgrove
# For SAGAN implementation
# Based on Non-local Neural Networks by Wang et al. https://arxiv.org/abs/1711.07971

from torch import nn
import torch.nn.functional as F
import torch
from spectral_normalization import SpectralNorm


class SelfAttentionPost(nn.Module):
    def __init__(self, input_size, attention_size):
        super(SelfAttentionPost, self).__init__()
        self.attention_size = attention_size
        self.gamma = nn.Parameter(torch.tensor(0.))
        self.h = SpectralNorm(nn.Conv2d(input_size, self.attention_size, 1, stride=1))
        self.i = SpectralNorm(nn.Conv2d(self.attention_size, input_size, 1, stride=1))

    def forward(self, x, att):
        width = x.size(2)
        height = x.size(3)
        m = x
        h = self.gamma * self.h(m)
        h = h.permute(0, 2, 3, 1).contiguous().view(-1, self.attention_size)
        h = torch.mm(att, h)
        h = h.view(-1, width, height, self.attention_size).permute(0, 3, 1, 2)
        m = self.i(h) + m
        return m



class SelfAttention(nn.Module):
    def __init__(self, input_size, attention_size):
        super(SelfAttention, self).__init__()
        self.attention_size = attention_size

        #attention layers
        self.f = SpectralNorm(nn.Conv2d(input_size, attention_size, 1, stride=1))
        self.g = SpectralNorm(nn.Conv2d(input_size, attention_size, 1, stride=1))
        self.input_size = input_size

    def forward(self, x):
        width = x.size(2)
        height = x.size(3)
        channels = x.size(1)
        m = x
        f = self.f(m)
        f = torch.transpose(f.view(-1, self.attention_size, width * height), 1, 2).contiguous()
        g = self.g(m)
        g = torch.transpose(g.view(-1, self.attention_size, width * height), 0, 1).contiguous()
        att = torch.mm(
            f.view(-1, self.attention_size),
            g.view(self.attention_size, -1))

        return F.softmax(att, 1)

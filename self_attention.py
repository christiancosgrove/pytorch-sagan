# Implementation of self-attention layer by Christian Cosgrove
# For SAGAN implementation
# Based on Non-local Neural Networks by Wang et al. https://arxiv.org/abs/1711.07971

class SelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(SelfAttention, self).__init__()
        self.attention_size = attention_size
        self.gamma = nn.Parameter(torch.tensor(0.))

        #attention layers
        self.f = SpectralNorm(nn.Conv2d(128, attention_size, 1, stride=1))
        self.g = SpectralNorm(nn.Conv2d(128, attention_size, 1, stride=1))
        self.h = SpectralNorm(nn.Conv2d(128, attention_size, 1, stride=1))
        self.i = SpectralNorm(nn.Conv2d(attention_size, 128, 1, stride=1))



    def forward(self, x):
        m = x
        f = self.f(m)
        g = self.g(m)
        att = torch.bmm(torch.transpose(f.view(-1, 8 * 8, self.attention_size), 1, 2), g.view(-1, 8 * 8, self.attention_size))

        h = self.gamma * self.h(m)
        h = att.view(-1, 8, 8, 1).expand_as(h) + h
        m = self.i(h) + m
        return m


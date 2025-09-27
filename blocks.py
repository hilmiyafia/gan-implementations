
import torch
from torch.nn.utils import spectral_norm

def norm(m, use_norm):
    return spectral_norm(m) if use_norm else m

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim, use_norm=False):
        super().__init__()
        self.layers = torch.nn.Sequential(
            norm(torch.nn.Conv2d(dim, dim, 3, 1, 1), use_norm),
            torch.nn.Tanh(),
            norm(torch.nn.Conv2d(dim, dim, 3, 1, 1), use_norm))
        self.se = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            norm(torch.nn.Conv2d(dim, dim // 2, 1), use_norm),
            torch.nn.LeakyReLU(),
            norm(torch.nn.Conv2d(dim // 2, dim, 1), use_norm),
            torch.nn.Sigmoid())
        self.act = torch.nn.Tanh()
    def forward(self, x):
        y = self.layers(x)
        return self.act(y * self.se(y) + x)
     
class DownBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, c=1, use_norm=False):
        super().__init__()
        self.layers = torch.nn.Sequential(
            *[ResidualBlock(in_dim, use_norm) for _ in range(c)],
            norm(torch.nn.Conv2d(in_dim, out_dim, 1), use_norm),
            torch.nn.MaxPool2d(2))
    def forward(self, x):
        return self.layers(x)

class UpBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, c=1, use_norm=False):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            norm(torch.nn.Conv2d(in_dim, out_dim, 1), use_norm),
            *[ResidualBlock(out_dim, use_norm) for _ in range(c)])
    def forward(self, x):
        return self.layers(x)

class StyleResidualBlock(torch.nn.Module):
    def __init__(self, dim, style, use_norm=False):
        super().__init__()
        self.style = style
        self.mapping = torch.nn.Sequential(
            norm(torch.nn.Conv2d(512, 512, 1), use_norm),
            torch.nn.LeakyReLU(),
            norm(torch.nn.Conv2d(512, 2 * dim, 1), use_norm))
        self.layers = torch.nn.Sequential(
            norm(torch.nn.Conv2d(dim, dim, 3, 1, 1), use_norm),
            torch.nn.Tanh(),
            norm(torch.nn.Conv2d(dim, dim, 3, 1, 1, bias=False), use_norm),
            torch.nn.InstanceNorm2d(dim))
        self.act = torch.nn.Tanh()
    def forward(self, x):
        u, s = self.mapping(self.style()).chunk(2, 1)
        return self.act(self.layers(x) * s + u + x)

class StyleUpBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, style, c=1, use_norm=False):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            norm(torch.nn.Conv2d(in_dim, out_dim, 1), use_norm),
            *[StyleResidualBlock(out_dim, style, use_norm) for _ in range(c)])
    def forward(self, x):
        return self.layers(x)

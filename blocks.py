
import torch
from torch.nn.utils import spectral_norm

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim, use_norm=False):
        super().__init__()
        norm = lambda x: spectral_norm(x) if use_norm else x
        self.layers = torch.nn.Sequential(
            norm(torch.nn.Conv2d(dim, dim, 3, padding=1)),
            torch.nn.Tanh(),
            norm(torch.nn.Conv2d(dim, dim, 3, padding=1)))
        self.activation = torch.nn.Tanh()
    def forward(self, x):
        return self.activation(self.layers(x) + x)

class DownBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, use_norm=False):
        super().__init__()
        norm = lambda x: spectral_norm(x) if use_norm else x
        self.layers = torch.nn.Sequential(
            ResidualBlock(in_dim, use_norm),
            norm(torch.nn.Conv2d(in_dim, out_dim, 1)),
            torch.nn.MaxPool2d(2))
    def forward(self, x):
        return self.layers(x)

class UpBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, use_norm=False):
        super().__init__()
        norm = lambda x: spectral_norm(x) if use_norm else x
        self.layers = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            norm(torch.nn.Conv2d(in_dim, out_dim, 1)),
            ResidualBlock(out_dim, use_norm))
    def forward(self, x):
        return self.layers(x)
    
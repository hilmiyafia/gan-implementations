
import torch
import torch.nn.init
from torch.nn.utils import spectral_norm

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

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
    
class StyleResidualBlock(torch.nn.Module):
    def __init__(self, dim, style, use_norm=False):
        super().__init__()
        norm = lambda x: spectral_norm(x) if use_norm else x
        self.layers = torch.nn.Sequential(
            norm(torch.nn.Conv2d(dim, dim, 3, padding=1)),
            torch.nn.Tanh(),
            norm(torch.nn.Conv2d(dim, dim, 3, padding=1, bias=False)))
        self.b_network = torch.nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.norm = torch.nn.InstanceNorm2d(dim)
        self.a_network = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 1),
            torch.nn.Tanh(),
            torch.nn.Conv2d(512, 2 * dim, 1))
        self.style = style
        self.activation = torch.nn.Tanh()
    def forward(self, x):
        u, s = self.a_network(self.style()).chunk(2, 1)
        y = self.layers(x) + torch.randn_like(x) * self.b_network
        return self.activation(self.norm(y) * s + u + x)

class DownBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, count=1, use_norm=False):
        super().__init__()
        norm = lambda x: spectral_norm(x) if use_norm else x
        self.layers = torch.nn.Sequential(
            *[ResidualBlock(in_dim, use_norm) for _ in range(count)],
            norm(torch.nn.Conv2d(in_dim, out_dim, 4, 2, 1)))
    def forward(self, x):
        return self.layers(x)

class UpBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, count=1, use_norm=False):
        super().__init__()
        norm = lambda x: spectral_norm(x) if use_norm else x
        self.layers = torch.nn.Sequential(
            norm(torch.nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1)),
            *[ResidualBlock(out_dim, use_norm) for _ in range(count)])
    def forward(self, x):
        return self.layers(x)

class StyleUpBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, style, count=1, use_norm=False):
        super().__init__()
        norm = lambda x: spectral_norm(x) if use_norm else x
        self.layers = torch.nn.Sequential(
            norm(torch.nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1)),
            *[StyleResidualBlock(out_dim, style, use_norm) for _ in range(count)])
    def forward(self, x):
        return self.layers(x)
    
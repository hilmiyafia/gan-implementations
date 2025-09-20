
import os
import torch
import torchvision
import pytorch_lightning
from blocks import DownBlock, StyleUpBlock
from torch.nn.utils import spectral_norm

class Generator(torch.nn.Module):
    def __init__(self, latent_dim, noise_dim):
        super().__init__()
        total_dim = latent_dim + noise_dim
        self.style = None
        get_style = lambda: self.style
        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(total_dim, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 512),
            torch.nn.Tanh(),
            torch.nn.Unflatten(1, (512, 1, 1)))
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 1),
            torch.nn.Tanh(),
            torch.nn.ConvTranspose2d(512, 512, 8, groups=512),
            StyleUpBlock(512, 256, get_style, 3),
            StyleUpBlock(256, 128, get_style, 3),
            StyleUpBlock(128, 64, get_style, 3),
            StyleUpBlock(64, 32, get_style, 3),
            torch.nn.Conv2d(32, 3, 1))
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(512, latent_dim, 4),
            torch.nn.Flatten(1))
    def forward(self, x):
        self.style = self.mapping(x)
        return self.layers(self.style)

class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            spectral_norm(torch.nn.Conv2d(3, 32, 7, 2, 3)),
            DownBlock(32, 64, 1, True),
            DownBlock(64, 128, 1, True),
            DownBlock(128, 256, 1, True),
            DownBlock(256, 512, 1, True))
        self.critic = torch.nn.Sequential(
            spectral_norm(torch.nn.Conv2d(512, 1, 4)),
            torch.nn.Flatten(1))
    def forward(self, x):
        features = self.features(x)
        return self.critic(features), features

class StyleGAN(pytorch_lightning.LightningModule):
    def __init__(self, model, critic, latent_dim, noise_dim):
        super().__init__()
        self.model = model
        self.critic = critic
        self.automatic_optimization = False
        self.total_dim = latent_dim + noise_dim
        self.latent_dim = latent_dim
    def configure_optimizers(self):
        return [
            torch.optim.AdamW(self.model.parameters(), 6e-5, [0.5, 0.9]),
            torch.optim.AdamW(self.critic.parameters(), 6e-5, [0.5, 0.9])]
    def training_step(self, batch, batch_idx):
        if batch_idx % 2 == 0:
            self.train_critic(batch)
        else:
            self.train_model(batch)
    def train_critic(self, batch):
        _, opt = self.optimizers()
        opt.zero_grad()
        noise = torch.randn(batch.shape[0], self.total_dim, device=batch.device)
        fake = self.model(noise)
        loss_fake = (self.critic(fake)[0] + 1).square().mean()
        loss_real = (self.critic(batch)[0] - 1).square().mean()
        self.log("c_fake", loss_fake, True)
        self.log("c_real", loss_real, True)
        self.manual_backward(loss_fake + loss_real)
        opt.step()
    def train_model(self, batch):
        opt, _ = self.optimizers()
        opt.zero_grad()
        noise = torch.randn(batch.shape[0], self.total_dim, device=batch.device)
        fake = self.model(noise)
        score, features = self.critic(fake)
        code = self.model.encoder(features)
        loss_fake = (score - 1).square().mean()
        loss_info = 40 * (code - noise[:, :self.latent_dim]).square().mean()
        self.log("m_fake", loss_fake, True)
        self.log("m_info", loss_info, True)
        self.manual_backward(loss_fake + loss_info)
        opt.step()
    def validation_step(self, batch, batch_index):
        output = self.model(batch[0])
        output = torchvision.utils.make_grid(output, 4, 0) / 2 + 0.5
        output = (output * 255).clamp(min=0, max=255).to(torch.uint8)
        os.makedirs("StyleGAN", exist_ok=True)
        path = f"StyleGAN/Image {self.global_step}.png"
        torchvision.io.write_png(output.cpu(), path)


import os
import torch
import torchvision
import pytorch_lightning
from blocks import ResidualBlock, DownBlock, UpBlock
from torch.nn.utils import spectral_norm

class Generator(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 1),
            DownBlock(32, 64),
            DownBlock(64, 128),
            DownBlock(128, 256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            torch.nn.Conv2d(32, 3, 1))
    def forward(self, x):
        return self.layers(x)

class Critic(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.layers = torch.nn.Sequential(
            spectral_norm(torch.nn.Conv2d(3, 32, 7, 2, 3)),
            torch.nn.LeakyReLU(),
            spectral_norm(torch.nn.Conv2d(32, 64, 4, 2, 1)),
            torch.nn.LeakyReLU(),
            spectral_norm(torch.nn.Conv2d(64, 128, 4, 2, 1)),
            torch.nn.LeakyReLU(),
            spectral_norm(torch.nn.Conv2d(128, 256, 4, 2, 1)),
            torch.nn.LeakyReLU(),
            spectral_norm(torch.nn.Conv2d(256, 512, 4, 2, 1)),
            torch.nn.LeakyReLU(),
            spectral_norm(torch.nn.Conv2d(512, 1, 4)),
            torch.nn.Flatten(1))
    def forward(self, x):
        return self.layers(x)

class Encoder(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 7, 2, 3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, 4, 2, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 128, 4, 2, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 256, 4, 2, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(256, 512, 4, 2, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(512, 8, 4),
            torch.nn.Flatten(1))
    def forward(self, x):
        return self.layers(x)

class TraVeLGAN(pytorch_lightning.LightningModule):
    def __init__(self, model, encoder, critic):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.critic = critic
        self.automatic_optimization = False
    def configure_optimizers(self):
        return [
            torch.optim.AdamW(self.model.parameters(), 6e-5, [0.5, 0.9]),
            torch.optim.AdamW(self.encoder.parameters(), 6e-5, [0.5, 0.9]),
            torch.optim.AdamW(self.critic.parameters(), 6e-5, [0.5, 0.9])]
    def training_step(self, batch, batch_idx):
        if batch_idx % 2 == 0:
            self.train_critic(batch)
        else:
            self.train_model(batch)
    def train_critic(self, batch):
        _, _, opt = self.optimizers()
        opt.zero_grad()
        fake = self.model(batch[0])
        loss_fake = (self.critic(fake) + 1).square().mean()
        loss_real = (self.critic(batch[1]) - 1).square().mean()
        self.log("c_fake", loss_fake, True)
        self.log("c_real", loss_real, True)
        self.manual_backward(loss_fake + loss_real)
        opt.step()
    def train_model(self, batch):
        opt_model, opt_encoder, _ = self.optimizers()
        opt_model.zero_grad()
        opt_encoder.zero_grad()
        fake = self.model(batch[0])
        fake_latents = self.encoder(fake).chunk(2, 0)
        real_latents = self.encoder(batch[0]).chunk(2, 0)
        fake_vector = fake_latents[0] - fake_latents[1]
        real_vector = real_latents[0] - real_latents[1]
        fake_mag = fake_vector.square().sum(1).sqrt()
        real_mag = real_vector.square().sum(1).sqrt()
        cos = (fake_vector * real_vector).sum(1) / (fake_mag * real_mag)
        loss_fake = (self.critic(fake) - 1).square().mean()
        loss_travel = (cos - 1).square().mean()
        loss_fake_mag = (1 - fake_mag).clamp(min=0).mean()
        loss_real_mag = (1 - real_mag).clamp(min=0).mean()
        self.log("m_fake", loss_fake, True)
        self.log("m_travel", loss_travel, True)
        self.log("m_fake_mag", loss_fake_mag, True)
        self.log("m_real_mag", loss_real_mag, True)
        self.manual_backward(loss_fake + loss_travel + loss_fake_mag + loss_real_mag)
        opt_model.step()
        opt_encoder.step()
    def validation_step(self, batch, batch_index):
        output = torch.stack((batch[0], self.model(batch[0])), 3)
        output = torchvision.utils.make_grid(output.flatten(2, 3), 4, 0) / 2 + 0.5
        output = (output * 255).clamp(min=0, max=255).to(torch.uint8)
        os.makedirs("TraVeLGAN", exist_ok=True)
        path = f"TraVeLGAN/Image {self.global_step}.png"
        torchvision.io.write_png(output.cpu(), path)

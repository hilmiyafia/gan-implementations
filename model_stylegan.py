
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
        self.styles = {}
        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(total_dim, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Unflatten(1, (512, 1, 1)))
        self.base = torch.nn.Parameter(torch.randn(1, 512, 8, 8))
        self.layers = torch.nn.Sequential(
            StyleUpBlock(512, 256, lambda: self.styles[0], 3),
            StyleUpBlock(256, 128, lambda: self.styles[1], 3),
            StyleUpBlock(128, 64, lambda: self.styles[2], 3),
            StyleUpBlock(64, 32, lambda: self.styles[3], 3),
            torch.nn.Conv2d(32, 3, 1))
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(512, latent_dim, 4),
            torch.nn.Flatten(1))
    def forward(self, x):
        style = self.mapping(x)
        for i in range(4): self.styles[i] = style
        return self.layers(self.base.repeat(x.shape[0], 1, 1, 1))
    def generate_with_styles(self, xs):
        for i in range(4): self.styles[i] = self.mapping(xs[i])
        return self.layers(self.base.repeat(xs[0].shape[0], 1, 1, 1))

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
        optimizers = [
            torch.optim.AdamW(self.model.parameters(), 6e-5, [0.5, 0.9]),
            torch.optim.AdamW(self.critic.parameters(), 6e-5, [0.5, 0.9])]
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(optimizers[0], 1, 1e-1, 750),
            torch.optim.lr_scheduler.LinearLR(optimizers[1], 1, 1e-1, 750)]
        return optimizers, schedulers
    def training_step(self, batch, batch_idx):
        model_opt, critic_opt = self.optimizers()
        if batch_idx % 2 == 0:
            self.train_critic(batch, critic_opt)
        else:
            self.train_model(batch, model_opt)
    def on_train_epoch_start(self):
        print(
            "\nEpoch:", self.current_epoch, 
            "LR:", self.lr_schedulers()[0].get_lr()[0])
    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        schedulers[0].step()
        schedulers[1].step()
    def train_critic(self, batch, opt):
        opt.zero_grad()
        noise = torch.randn(batch.shape[0], self.total_dim).to(batch.device)
        fake = self.model(noise)
        loss_fake = (self.critic(fake)[0] + 1).square().mean()
        loss_real = (self.critic(batch)[0] - 1).square().mean()
        self.log("c_fake", loss_fake, True)
        self.log("c_real", loss_real, True)
        self.manual_backward(loss_fake + loss_real)
        opt.step()
    def train_model(self, batch, opt):
        opt.zero_grad()
        noise = torch.randn(batch.shape[0], self.total_dim).to(batch.device)
        fake = self.model(noise)
        score, features = self.critic(fake)
        code = self.model.encoder(features)
        loss_fake = score.square().mean()
        loss_info = 50 * (code - noise[:, :self.latent_dim]).square().mean()
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

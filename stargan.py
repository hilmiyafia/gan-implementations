
import os
import torch
import torchvision
import pytorch_lightning
from blocks import ResidualBlock, DownBlock, UpBlock
from torch.nn.utils import spectral_norm

class Generator(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding(count, 3),
            torch.nn.Unflatten(1, (3, 1, 1)))
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(6, 32, 1),
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
    def forward(self, x, c):
        e = torch.ones_like(x) * self.embedding(c)
        return self.layers(torch.concat((x, e), 1))

class Critic(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.features = torch.nn.Sequential(
            spectral_norm(torch.nn.Conv2d(3, 32, 7, 2, 3)),
            torch.nn.LeakyReLU(),
            spectral_norm(torch.nn.Conv2d(32, 64, 4, 2, 1)),
            torch.nn.LeakyReLU(),
            spectral_norm(torch.nn.Conv2d(64, 128, 4, 2, 1)),
            torch.nn.LeakyReLU(),
            spectral_norm(torch.nn.Conv2d(128, 256, 4, 2, 1)),
            torch.nn.LeakyReLU(),
            spectral_norm(torch.nn.Conv2d(256, 512, 4, 2, 1)),
            torch.nn.LeakyReLU())
        self.critic = torch.nn.Sequential(
            spectral_norm(torch.nn.Conv2d(512, 1, 4)),
            torch.nn.Flatten(1))
        self.domain = torch.nn.Sequential(
            spectral_norm(torch.nn.Conv2d(512, count, 4)),
            torch.nn.Flatten(1))
    def forward(self, x):
        z = self.features(x)
        return self.critic(z), self.domain(z)

class StarGAN(pytorch_lightning.LightningModule):
    def __init__(self, model, critic):
        super().__init__()
        self.model = model
        self.critic = critic
        self.automatic_optimization = False
        self.loss = torch.nn.CrossEntropyLoss()
    def configure_optimizers(self):
        return [
            torch.optim.AdamW(self.model.parameters(), 6e-5, [0.5, 0.9]),
            torch.optim.AdamW(self.critic.parameters(), 6e-5, [0.5, 0.9])]
    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            device = batch[0].device
            c_source = [i for i, b in enumerate(batch) for _ in range(b.shape[0])]
            c_source = torch.LongTensor(c_source).to(device)
            c_target = torch.randint(0, len(batch) - 1, c_source.shape[0], device=device)
            c_target[c_target >= c_source] += 1
            batch = torch.concat(batch)
        if batch_idx % 2 == 0:
            self.train_critic(batch, c_source, c_target)
        else:
            self.train_model(batch, c_source, c_target)
    def train_critic(self, batch, c_source, c_target):
        _, opt = self.optimizers()
        opt.zero_grad()
        fake = self.model(batch, c_target)
        score_fake, _ = self.critic(fake)
        score_real, score_class = self.critic(batch)
        loss_fake = (score_fake + 1).square().mean()
        loss_real = (score_real - 1).square().mean()
        loss_class = self.loss(score_class, c_source)
        self.log("c_fake", loss_fake, True)
        self.log("c_real", loss_real, True)
        self.log("c_class", loss_class, True)
        self.manual_backward(loss_fake + loss_real + loss_class)
        opt.step()
    def train_model(self, batch, c_source, c_target):
        opt, _ = self.optimizers()
        opt.zero_grad()
        fake = self.model(batch, c_target)
        output = self.model(fake, c_source)
        score_fake, score_class = self.critic(fake)
        loss_fake = (score_fake - 1).square().mean()
        loss_class = self.loss(score_class, c_target)
        loss_output = 100 * (output - batch).square().mean()
        self.log("m_fake", loss_fake, True)
        self.log("m_class", loss_class, True)
        self.log("m_output", loss_output, True)
        self.manual_backward(loss_fake + loss_class + loss_output)
        opt.step()
    def validation_step(self, batch, batch_index):
        device = batch[0].device
        c_target = torch.ones(batch[0].shape[0], device=device, dtype=torch.int64)
        output = torch.stack((batch[0], self.model(batch[0], c_target)), 3)
        output = torchvision.utils.make_grid(output.flatten(2, 3), 4, 0) / 2 + 0.5
        output = (output * 255).clamp(min=0, max=255).to(torch.uint8)
        os.makedirs("StarGAN", exist_ok=True)
        path = f"StarGAN/Image {self.global_step}.png"
        torchvision.io.write_png(output.cpu(), path)


import os
import torch
import pytorch_lightning
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from utils import get_dataset, get_dataloader
from model_infogan import Generator, Critic, InfoGAN
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    EPOCHS = 100
    VAL_SIZE = 16
    VAL_INTERVAL = 100
    BATCH_SIZE = 8
    LATENT_DIM = 8
    NOISE_DIM = 16

    dataset, dataloader = get_dataset("../dataset/celeb", BATCH_SIZE)
    generator = torch.Generator()
    generator.manual_seed(0)
    val_dataset = torch.utils.data.TensorDataset(
        torch.randn(VAL_SIZE, LATENT_DIM + NOISE_DIM, generator=generator))
    val_dataloader = get_dataloader(val_dataset, VAL_SIZE, True, False)[1]

    model = Generator(LATENT_DIM, NOISE_DIM)
    critic = Critic()
    adversarial = InfoGAN(model, critic, LATENT_DIM, NOISE_DIM)

    trainer = pytorch_lightning.Trainer(
        max_epochs=EPOCHS,
        val_check_interval=VAL_INTERVAL,
        check_val_every_n_epoch=None,
        callbacks=[ModelCheckpoint(save_on_train_epoch_end=True)])
    checkpoint = None
    
    # base_path = "lightning_logs/version_0/checkpoints/"
    # checkpoint = base_path + os.listdir(base_path)[0]

    trainer.fit(
        model=adversarial, 
        train_dataloaders=dataloader, 
        val_dataloaders=val_dataloader, 
        ckpt_path=checkpoint)

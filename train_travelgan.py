
import os
import torch
import pytorch_lightning
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from utils import get_dataset, get_dataloader
from travelgan import Generator, Critic, Encoder, TraVeLGAN
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    EPOCHS = 100
    VAL_SIZE = 8
    VAL_INTERVAL = 100
    BATCH_SIZE = 8
    dataset_a, dataloader_a = get_dataset("../dataset/cats", BATCH_SIZE)
    dataset_b, dataloader_b = get_dataset("../dataset/dogs", BATCH_SIZE)
    dataloader = CombinedLoader([dataloader_a, dataloader_b], "max_size_cycle")
    val_dataset = torch.utils.data.TensorDataset(
        torch.stack([dataset_a.get_unflipped(i * 20) for i in range(VAL_SIZE)]))
    val_dataloader = get_dataloader(val_dataset, VAL_SIZE, True, False)[1]

    model = Generator()
    encoder = Encoder()
    critic = Critic()
    adversarial = TraVeLGAN(model, encoder, critic)

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


import os
import torch
import torchvision
import torchvision.transforms.v2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.paths = [f"{path}/{file}" for file in os.listdir(path)]
        self.paths.sort()
        self.norm = torchvision.transforms.v2.ToDtype(torch.float32, True)
        self.flip = torchvision.transforms.v2.RandomHorizontalFlip()
    def __len__(self):
        return len(self.paths)
    def resize(self, x):
        return torch.nn.functional.interpolate(
            x[None], (128, 128), mode="area")[0]
    def __getitem__(self, index):
        image = torchvision.io.read_image(self.paths[index])
        return self.resize(self.flip(self.norm(image)) * 2 - 1)
    def get_unflipped(self, index):
        image = torchvision.io.read_image(self.paths[index])
        return self.resize(self.norm(image) * 2 - 1)

def get_dataset(path, batch_size, drop_last=True):
    return get_dataloader(Dataset(path), batch_size, drop_last)

def get_dataloader(dataset, batch_size, drop_last=True, shuffle=True):
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4, 
        persistent_workers=True, 
        drop_last=drop_last)
    return dataset, dataloader

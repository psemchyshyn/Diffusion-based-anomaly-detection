from torch.utils import data
from torchvision import transforms
from pathlib import Path
import torch
from PIL import Image
import os


class TrainDiffusionDataset(data.Dataset):
    def __init__(self, folder_normal, category="bottle", image_size=256, channels=3, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder_normal = folder_normal

        self.image_size = image_size
        self.channels = channels
        self.category = category
        self.paths = list(filter(lambda x: self.category in x, os.listdir(folder_normal)))
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path = os.path.join(self.folder_normal, self.paths[index])

        img = Image.open(image_path).convert("RGB")

        return {"image": self.transform(img),
                "label": torch.tensor([1]),
                "path": image_path}


class TestDiffusionDataset(data.Dataset):
    def __init__(self, folder_normal, folder_mask, folder_reconstruction, category="bottle", image_size=256, channels=3, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder_normal = folder_normal
        self.folder_reconstruction = folder_reconstruction
        self.folder_mask = folder_mask
        self.image_size = image_size
        self.channels = channels
        self.category = category
        self.paths = list(filter(lambda x: self.category in x, os.listdir(folder_normal)))
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path = os.path.join(self.folder_normal, self.paths[index])
        mask_path = os.path.join(self.folder_mask, self.paths[index])
        rec_path = os.path.join(self.folder_reconstruction, self.paths[index])

        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = self.transform(mask)
        rec = Image.open(rec_path).convert("RGB")
        label = (mask.sum() > 0).int()
        return {"image": self.transform(img),
                "mask": mask,
                "reconstruction": self.transform(rec),
                "label": label,
                "path": image_path}

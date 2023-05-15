import os
from torch.utils import data
from torchvision import transforms
from PIL import Image


class DiffusionDataset(data.Dataset):
    def __init__(self, folder_normal, folder_mask, folder_reconstruction,  image_size, channels=3, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder_normal = folder_normal
        self.folder_reconstruction = folder_reconstruction
        self.folder_mask = folder_mask
        self.image_size = image_size
        self.channels = channels
        self.paths = os.listdir(folder_normal)
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

from torch.utils import data
from torchvision import transforms
from pathlib import Path
from PIL import Image


class DiffusionDataset(data.Dataset):
    def __init__(self, folder, image_size, channels=3, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        if self.channels == 1:
            img = Image.open(path).convert("L")
        else:
            img = Image.open(path).convert("RGB")
        return {"path": str(path), "image": self.transform(img)}

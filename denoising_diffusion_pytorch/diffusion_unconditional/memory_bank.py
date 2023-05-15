import torch
import clip
import numpy as np
import os
from torch import Tensor
from typing import Tuple
from torch.utils import data
from tqdm import tqdm
from pathlib import Path
from PIL import Image


class MemoryBankDataset(data.Dataset):
    def __init__(self, folder, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        batch = {}
        path = self.paths[index]
        batch["path"] = str(path)
        return batch


class MemoryBank:
    def __init__(self, folder, batch_size=32):
        # memory bank
        self.folder = folder
        # % of items in folder in a memory bank
        # self.coreset_ratio = coreset_ratio
        self.memory_ds = MemoryBankDataset(self.folder)
        # batch_size
        self.batch_size = batch_size
        # memory bank
        self.memory_bank = []
        self.names_mapping = []


        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor, self.feature_extractor_preprocess = clip.load("ViT-B/32", device=self.device)

    def get_features(self, batch):
        image = [self.feature_extractor_preprocess(el).to(self.device).unsqueeze(0) for el in batch]

        image = torch.vstack(image)

        with torch.no_grad():
            image_features = self.feature_extractor.encode_image(image)
            return image_features.float()


    def subsample_embedding(self, embedding) -> None:
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """
        self.memory_bank = embedding

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int) -> Tuple[Tensor, Tensor]:
        """Nearest Neighbours using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """

        distances = torch.cdist(embedding, self.memory_bank, p=2.0)  # euclidean norm
        patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """Reshape Embedding.


        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        batch_size = embedding.size(0)
        embedding = embedding.reshape(batch_size, -1)
        return embedding

    def forward(self, batch):
        batch_embedding = self.reshape_embedding(self.get_features(batch))
        self.memory_bank.append(batch_embedding)

    def compute_memory_bank(self):
        dl = data.DataLoader(self.memory_ds, batch_size=self.batch_size, pin_memory=True)

        for batch in tqdm(dl):
            # print(batch["image"].shape, batch["path"].shape)
            batch["image"] = [Image.open(path)for path in batch["path"]]
            self.forward(batch["image"])
            self.names_mapping.extend(batch["path"])

        self.memory_bank = torch.cat(self.memory_bank, dim=0)
        self.names_mapping = np.array(self.names_mapping)
        self.subsample_embedding(self.memory_bank)

    def select(self, batch: torch.Tensor, n=1):
        batch_embedding = self.reshape_embedding(self.get_features(batch))
        distances, locations = self.nearest_neighbors(batch_embedding, n)
        locations = locations.squeeze(1).cpu().numpy()
        closest_features = self.memory_bank[locations]
        closest_names = self.names_mapping[locations]
        distances = torch.mean(distances, 1)
        return closest_features, closest_names, distances

    def test_closest(self, folder, out_folder="temp", num=10):
        test_ds = MemoryBankDataset(folder)
        dl = data.DataLoader(test_ds, batch_size=1, pin_memory=True)

        for i, batch in enumerate(dl):
            batch["image"] = [Image.open(path) for path in batch["path"]]
            closest_features, closest_names = self.select(batch["image"])
            path_original = batch["path"][0]
            path_closest = closest_names[0]
            out = path_original.split(os.sep)[-1]
            self.save_images([str(path_original), path_closest], out=f"{out_folder}/closest-to-{out}")

            if i > num:
                break

    @staticmethod
    def save_images(paths, out="test.png"):
        images = [Image.open(x) for x in paths]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save(out)

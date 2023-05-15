import os
from embeddings.coreset import KCenterGreedy
from embeddings.dataset import TestDiffusionDataset, TrainDiffusionDataset
from embeddings.anomaly_map import AnomalyMapGenerator
from denoising_diffusion_pytorch.diffusion_unconditional.lightning import LitModel
import numpy as np
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from segmentation.utils import get_auc, save_grids


class EmbeddingsLit(pl.LightningModule):
    def __init__(
        self,
        conf_training,
        conf_data,
        conf_anomaly,
    ):
        super().__init__()
        self.save_hyperparameters()
        conf_training, conf_data, conf_anomaly = self.hparams["conf_training"], self.hparams["conf_data"], self.hparams["conf_anomaly"]

        self.reconstruction_model = LitModel.load_from_checkpoint(conf_training["reconstruction_path"])
        self.model = self.reconstruction_model.ema_model.model
        self.batch_size = conf_training["batch_size"]
        self.image_size = conf_training["image_size"]
        self.channels = conf_training["channels"]


        self.layers_enc = conf_training["layers_enc"]
        self.layers_dec = conf_training["layers_dec"]
        self.num_neighbors = conf_training["num_neighbours"]
        self.sampling_ratio = conf_training["sampling_ratio"]

        # self.train_folder = conf_data["train_folder"]
        # self.val_folder = conf_data["val_folder"]
        self.category = conf_data["category"]
        self.test_output_folder = conf_training["test_output_dir"]
        self.train_ds = TrainDiffusionDataset(conf_data["train_folder_images"], self.category, self.image_size, self.channels)
        self.val_ds = TrainDiffusionDataset(conf_data["val_folder_images"], self.category, self.image_size, self.channels)
        self.test_ds = TestDiffusionDataset(conf_data["test_folder_images"], conf_data["test_folder_mask"], conf_data["test_folder_rec"], self.category, self.image_size, self.channels)

        self.anomaly_map_generator = AnomalyMapGenerator(input_size=[self.image_size, self.image_size])
        self.memory_bank = []

        # self.register_buffer("memory", torch.Tensor())

    def generate_embedding(self, fe, fd):
        """Generate embedding from hierarchical feature map.
        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: dict[str:Tensor]:
        Returns:
            Embedding vector
        """


        embeddings = fe[self.layers_enc[-1]]
        for layer in self.layers_enc[:-1]:
            layer_embedding = fe[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        for layer in self.layers_dec:
            layer_embedding = fd[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding):
        """Reshape Embedding.
        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]
        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.
        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding

    def subsample_embedding(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """

        # Coreset Subsampling
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset()
        self.memory_bank = coreset

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int):
        """Nearest Neighbours using brute force method and euclidean norm.
        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at
        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        distances = torch.cdist(embedding, self.memory_bank, p=2.0)  # euclidean norm
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def compute_anomaly_score(self, patch_scores: torch.Tensor, locations: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """Compute Image-Level Anomaly Score.
        Args:
            patch_scores (Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores
        Returns:
            Tensor: Image-level anomaly scores
        """

        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        _, support_samples = self.nearest_neighbors(nn_sample, n_neighbors=self.num_neighbors)
        # 4. Find the distance of the patch features to each of the support samples
        distances = torch.cdist(max_patches_features.unsqueeze(1), self.memory_bank[support_samples], p=2.0)
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        score = weights * score  # s in the paper
        return score

    def extract_features(self, batch):
        t = torch.full((batch.shape[0],), 0, device=batch.device, dtype=torch.long)
        with torch.no_grad():
            _, fe, fd = self.model(batch, t, return_features=True)
            print("Encoder shapes")

            for i in fe:
                print(i.shape)

            print("Decoder shape")
            for i in fd:
                print(i.shape)
            return fe, fd

    def forward(self, batch):
        fe, fd = self.extract_features(batch["image"])
        fd_embeddings = self.generate_embedding(fe, fd)
        return fd_embeddings


    def training_step(self, batch, batch_idx):
        self.model.eval()
        embs = self(batch)
        embs = self.reshape_embedding(embs)
        self.memory_bank.append(embs)
        return None

    def on_train_end(self) -> None:
        self.memory_bank = torch.cat(self.memory_bank, dim=0)
        print(self.memory_bank.shape)
        # self.memory_bank = self.memory_bank[torch.randint(0, len(self.memory_bank), size=(len(self.memory_bank) // 2, ))]
        self.subsample_embedding(self.memory_bank, self.sampling_ratio)
        self.memory_bank = self.memory_bank
        # print(self.memory_bank.shape)

    def on_test_start(self) -> None:
        self.test_preds = {"image": [], "reconstruction": [], "pred_mask": [], "mask": [], 'label': [], "anomaly_score": []}
        self.names = []

    def test_step(self, batch, batch_idx):
        embedding = self(batch)
        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
        # reshape to batch dimension
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        # compute anomaly score

        anomaly_score = self.compute_anomaly_score(patch_scores.cuda(), locations.cuda(), embedding.cuda())
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))

        print("Patchcore shape")
        print(patch_scores.shape)
        anomaly_map = self.anomaly_map_generator(patch_scores)

        normalizer = transforms.Normalize(mean=[0.5], std=[0.5])
        anomaly_map = normalizer(anomaly_map)

        self.test_preds["image"].append(batch["image"])
        self.test_preds["reconstruction"].append(batch["reconstruction"])
        self.test_preds["pred_mask"].append(anomaly_map)
        self.test_preds["mask"].append(batch["mask"])
        self.test_preds["label"].append(batch["label"])
        self.test_preds["anomaly_score"].append(anomaly_score)
        self.names.extend([path.split(os.sep)[-1] for path in batch["path"]])

    def on_test_end(self) -> None:
        test_dct = {key: torch.cat(val, dim=0) for key, val in self.test_preds.items()}

        names = [f"{round(score.item(), 3)}-{name}" for score, name in zip(test_dct["anomaly_score"], self.names)]

        true = np.array([label.item() for label in test_dct["label"]])
        preds = np.array([score.item() for score in test_dct["anomaly_score"]])
        auc = get_auc(preds, true)

        save_grids(self.test_output_folder,
                   test_dct["image"],
                   test_dct["reconstruction"],
                   test_dct["pred_mask"],
                   test_dct["mask"],
                   names=names,
                   grid_shape=(2, 2))

        print(f"The AUC for the class {self.category} is {auc}")
    # def on_validation_end(self) -> None:
    #     normalizer = transforms.Normalize(mean=[0.5],std=[0.5])
    #     self.vis_bank = torch.cat(self.vis_bank, dim=0)
    #     self.vis_bank = torch.mean(self.vis_bank, dim=1).unsqueeze(1)[:20]
    #     print(self.vis_bank.shape)
    #     self.vis_bank = normalizer(self.vis_bank)
    #     names = self.names[:20]
    #     save_grids(self.val_output_folder,
    #                self.vis_bank,
    #                names=names,
    #                grid_shape=(1, 1))
    #
    #
    # def on_test_start(self) -> None:
    #     self.test_preds = {"image": [], "reconstruction": [], "pred_mask": [], "mask": [], 'label': []}
    #     self.names = []
    #
    # def test_step(self, batch, batch_idx):
    #     predictions = self(batch)
    #     image_level, _ = predictions.view(predictions.size(0), -1).max(dim=-1)
    #     predictions = (predictions > 0.5).float()
    #
    #     self.test_preds["image"].append(batch["image"])
    #     self.test_preds["reconstruction"].append(batch["reconstruction"])
    #     self.test_preds["pred_mask"].append(predictions)
    #     self.test_preds["mask"].append(batch["mask"])
    #     self.test_preds["label"].append(batch["label"])
    #
    #     self.names.extend([path.split(os.sep)[-1] for path in batch["path"]])
    #

    # def on_test_end(self) -> None:
    #     test_dct = {key: torch.cat(val, dim=0) for key, val in self.test_preds.items()}
    #     image_level, _ = test_dct["pred_mask"].view(test_dct["pred_mask"].size(0), -1).max(dim=-1)
    #
    #     names = [f"{round(score.item(), 3)}-{name}" for score, name in zip(image_level, self.names)]
    #     classes = [name.split("-")[0] for name in self.names]
    #
    #     true = np.array([label.item() for label in test_dct["label"]])
    #     preds = np.array([score.item() for score in image_level])
    #     image_aucs = {}
    #     for clas in set(classes):
    #         idx = [clas == image_clas for image_clas in classes]
    #         auc = get_auc(preds[idx], true[idx])
    #         image_aucs[clas] = auc
    #
    #     save_grids(self.test_output_folder,
    #                test_dct["image"],
    #                test_dct["reconstruction"],
    #                test_dct["pred_mask"],
    #                test_dct["mask"],
    #                names=names,
    #                grid_shape=(2, 2))
    #     print(image_aucs)


    def train_dataloader(self):
        return data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return data.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    def configure_optimizers(self):
        pass

import os
from segmentation.dataset import DiffusionDataset
from segmentation.augmenter import Augmenter
from segmentation.discriminative import DiscriminativeSubNetwork
from denoising_diffusion_pytorch.diffusion_unconditional.lightning import LitModel
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils import data
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
from segmentation.utils import get_auc, save_grids


class SegLit(pl.LightningModule):
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
        self.reconstruction_model = self.reconstruction_model.ema_model
        self.reconstruction_model.eval()
        self.batch_size = conf_training["batch_size"]
        self.lr = conf_training["lr"]
        self.in_channels = conf_training["in_channels"]
        self.out_channels = conf_training["out_channels"]
        self.image_size = conf_training["image_size"]
        self.channels = conf_training["channels"]
        self.noise_amount = conf_training["noise_amount"]
        self.log_every_n_epoch = conf_training["log_every_n_epoch"]
        self.num_imgs_to_log = conf_training["num_imgs_to_log"]


        self.augmenter = Augmenter(**conf_anomaly)

        self.segmentation_network = DiscriminativeSubNetwork(self.in_channels, self.out_channels)
        self.loss = nn.CrossEntropyLoss()
        #
        # self.train_folder = conf_data["train_folder"]
        # self.val_folder = conf_data["val_folder"]
        self.test_output_folder = conf_training["test_output_dir"]
        self.train_ds = DiffusionDataset(conf_data["train_folder_images"], conf_data["train_folder_mask"], conf_data["train_folder_rec"], self.image_size, self.channels)
        self.val_ds = DiffusionDataset(conf_data["val_folder_images"], conf_data["val_folder_mask"], conf_data["val_folder_rec"], self.image_size, self.channels)
        self.test_ds = DiffusionDataset(conf_data["test_folder_images"], conf_data["test_folder_mask"], conf_data["test_folder_rec"], self.image_size, self.channels)

    def get_reconstruction(self, batch):
        # t = torch.full((batch.shape[0],), self.noise_amount, device=batch.device, dtype=torch.long)
        # noisy = self.reconstruction_model.q_sample(batch, t)

        # _, rec, _ = self.reconstruction_model.p_sample(noisy, self.noise_amount)
        rec = self.reconstruction_model.reconstruct_simple(batch, self.noise_amount, 3)
        return rec

    def forward(self, batch):
        # reconstruction = self.get_reconstruction(batch["image"])
        concatenated_inputs = torch.cat([batch["image"], batch["reconstruction"]], axis=1)
        prediction = self.segmentation_network(concatenated_inputs)
        # prediction = torch.nn.functional.softmax(prediction, dim=1)[:, 1, ...].unsqueeze(1)
        prediction = prediction[:, 1, ...].unsqueeze(1)

        return prediction

    def shared_step(self, batch, mode):
        # reconstruction = self.get_reconstruction(batch["image"])

        # augmented, mask = self.augmenter.augment_batch(batch["image"])
        # mask = mask.long()
        concatenated_inputs = torch.cat([batch["image"], batch["reconstruction"]], axis=1)
        prediction = self.segmentation_network(concatenated_inputs)

        loss = self.loss(prediction, batch["mask"].squeeze(1).long())
        prediction = torch.nn.functional.softmax(prediction, dim=1)
        prediction = prediction[:, 1, ...].unsqueeze(1)

        pred_class = (prediction > 0.5)
        anomaly_map = prediction

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True)
        return loss, batch["image"], batch["reconstruction"], anomaly_map, pred_class.float(), batch["mask"].float()

    def training_step(self, batch, batch_idx):
        loss, *_ = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, augmented, reconstruction, anomaly_map, pred_class, mask = self.shared_step(batch, "val")

        if self.current_epoch % self.log_every_n_epoch == 0:
            for i in range(1):
                imgs = map(TF.to_pil_image, [augmented[i], reconstruction[i], mask[i], anomaly_map[i], pred_class[i]])
                self.logger.log_image(key=f"train-{self.current_epoch}", images=list(imgs), caption=["image", "reconstruction", "mask", "anomaly_map", "pred_mask"])

        return loss

    def on_test_start(self) -> None:
        self.test_preds = {"image": [], "reconstruction": [], "pred_mask": [], "mask": [], 'label': []}
        self.names = []

    def test_step(self, batch, batch_idx):
        predictions = self(batch)
        image_level, _ = predictions.view(predictions.size(0), -1).max(dim=-1)
        predictions = (predictions > 0.5).float()

        self.test_preds["image"].append(batch["image"])
        self.test_preds["reconstruction"].append(batch["reconstruction"])
        self.test_preds["pred_mask"].append(predictions)
        self.test_preds["mask"].append(batch["mask"])
        self.test_preds["label"].append(batch["label"])

        self.names.extend([path.split(os.sep)[-1] for path in batch["path"]])


    def on_test_end(self) -> None:
        test_dct = {key: torch.cat(val, dim=0) for key, val in self.test_preds.items()}
        image_level = test_dct["pred_mask"].view(test_dct["pred_mask"].size(0), -1).max(dim=-1)[0] / (test_dct["pred_mask"].view(test_dct["pred_mask"].size(0), -1).min(dim=-1)[0] + 0.001)

        names = [f"{round(score.item(), 3)}-{name}" for score, name in zip(image_level, self.names)]
        classes = [name.split("-")[0] for name in self.names]

        true = np.array([label.item() for label in test_dct["label"]])
        preds = np.array([score.item() for score in image_level])
        image_aucs = {}
        for clas in set(classes):
            idx = [clas == image_clas for image_clas in classes]
            auc = get_auc(preds[idx], true[idx])
            image_aucs[clas] = auc

        save_grids(self.test_output_folder,
                   test_dct["image"],
                   test_dct["reconstruction"],
                   test_dct["pred_mask"],
                   test_dct["mask"],
                   names=names,
                   grid_shape=(2, 2))
        print(image_aucs)


    def train_dataloader(self):
        return data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return data.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.segmentation_network.parameters(), lr=self.lr)
        return optimizer

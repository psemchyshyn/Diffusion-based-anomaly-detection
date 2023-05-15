from classification.dataset import DiffusionDataset
from classification.discriminative import DiscriminativeSubNetwork
from segmentation.augmenter import Augmenter
from denoising_diffusion_pytorch.diffusion_unconditional.lightning import LitModel
from torch.utils import data
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms.functional as TF


class ClassifierLit(pl.LightningModule):
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

        self.augmenter = Augmenter(**conf_anomaly)

        # self.segmentation_network = DiscriminativeSubNetwork(self.in_channels, self.out_channels)

        self.classifier_network = DiscriminativeSubNetwork()
        self.loss = nn.CrossEntropyLoss()


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
        concatenated_inputs = torch.cat([batch["image"], batch["reconstruction"]], axis=1)
        prediction = self.classifier_network(batch["image"] + batch["reconstruction"])
        prediction = prediction[:, 1, ...]
        return prediction

    def shared_step(self, batch, mode):
        concatenated_inputs = torch.cat([batch["image"], batch["reconstruction"]], axis=1)
        prediction = self.classifier_network(batch["image"] + batch["reconstruction"])
        loss = self.loss(prediction, batch["label"].long())
        prediction = prediction[:, 1, ...]

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True)
        return loss, batch["image"], batch["reconstruction"], prediction.float(), batch["label"].float()

    def training_step(self, batch, batch_idx):
        loss, *_ = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, augmented, reconstruction, pred_anomaly, gt = self.shared_step(batch, "val")

        if self.current_epoch % self.log_every_n_epoch == 0:
            for i in range(1):
                anomaly_level = pred_anomaly[i].item()
                gt_item = gt[i].item()
                imgs = map(TF.to_pil_image, [augmented[i], reconstruction[i]])
                self.logger.log_image(key=f"train-{self.current_epoch}", images=list(imgs), caption=[f"image-{gt_item}", f"reconstruction-{anomaly_level}"])

        return loss


    def train_dataloader(self):
        return data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return data.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier_network.parameters(), lr=self.lr)
        return optimizer

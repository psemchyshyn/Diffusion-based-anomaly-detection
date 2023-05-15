import os
from denoising_diffusion_pytorch.dataset import DiffusionDataset
from denoising_diffusion_pytorch.unet import Unet
from denoising_diffusion_pytorch.gaussian_diffusion import GaussianDiffusion
from torch.utils import data
from denoising_diffusion_pytorch.utils import num_to_groups
from denoising_diffusion_pytorch.memory_bank import MemoryBank
import torch
import pytorch_lightning as pl
from PIL import Image
import torchvision.utils as utils
import copy



class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class LitModel(pl.LightningModule):
    def __init__(
        self,
        conf_model,
        conf_training,
        conf_data,
    ):
        super().__init__()
        self.save_hyperparameters()
        conf_model, conf_training, conf_data = self.hparams["conf_model"], self.hparams["conf_training"], self.hparams["conf_data"]

        backbone = Unet(**conf_model["denoising_fn"])
        model = GaussianDiffusion(backbone, **conf_model["diffusion"])
        self.model = model
        self.ema_model = copy.deepcopy(self.model)

        self.ema_decay = conf_training["ema_decay"]
        self.update_ema_every = conf_training["update_ema_every"]
        self.step_start_ema = conf_training["step_start_ema"]
        self.ema = EMA(self.ema_decay)

        self.batch_size = conf_training["batch_size"]
        self.acc_grad_batches = conf_training["acc_grad_batches"]
        self.lr = conf_training["lr"]
        self.milestone_results_dir = conf_training["milestone_results_dir"]
        self.generate_samples_every_n_epochs = conf_training["generate_samples_every_n_epochs"]
        os.makedirs(self.milestone_results_dir, exist_ok=True)

        self.image_size = conf_model["diffusion"]["image_size"]
        self.channels = conf_model["denoising_fn"]["channels"]
        self.train_folder = conf_data["train_folder"]
        self.val_folder = conf_data["val_folder"]
        self.train_ds = DiffusionDataset(self.train_folder, self.image_size, self.channels)
        self.val_ds = DiffusionDataset(self.val_folder, self.image_size, self.channels)
        self.register_buffer("memory_bank", torch.rand((len(os.listdir(conf_training["embedding_folder"])), 512)))

        self.embedding_folder = conf_training["embedding_folder"]
        self.storage = MemoryBank(self.embedding_folder)

    def setup(self, stage):
        super().setup(stage)
        if stage == "fit":
            self.storage.compute_memory_bank()
        self.register_buffer("memory_bank", self.storage.memory_bank)


    def step_ema(self):
        if self.global_step < self.step_start_ema:
            self.ema_model.load_state_dict(self.model.state_dict())
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def shared_step(self, batch, mode):
        conditions, _ = self.storage.select([Image.open(path) for path in batch["path"]])
        loss = self.model(batch["image"], conditions)
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True)
        return loss


    def training_step(self, batch, batch_idx):
        if self.global_step % self.update_ema_every == 0:
            self.step_ema()
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def on_validation_epoch_end(self):
        if self.current_epoch % self.generate_samples_every_n_epochs != 0:
            return

        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample(embeddings=self.memory_bank[torch.randint(len(self.memory_bank), (n,)), :], batch_size=n), batches))
        all_images = torch.cat(all_images_list, dim=0)
        utils.save_image(all_images, f'{self.milestone_results_dir}/sample-{self.current_epoch}.png', nrow = 6)

    def train_dataloader(self):
        return data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

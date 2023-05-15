import os
from PIL import Image


import os
import torch
import numpy as np
import torch.nn.functional as F
from utils import save_grids
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from classification.dataset import DiffusionDataset
from denoising_diffusion_pytorch.diffusion_unconditional.lightning import LitModel
import tqdm


class Predictor:
    def __init__(self, diffusion_model, image_size=256, channels=3, batch_size=8):
        self.diffusion_model = diffusion_model
        self.unet_model = diffusion_model.model
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size


    def extract_features(self, batch):
        t = torch.full((batch.shape[0],), 500, device=batch.device, dtype=torch.long)
        with torch.no_grad():
            _, fe, fd = self.unet_model(batch, t, return_features=True)

        return fe, fd


    def calculate_difference(self, orig, rec):
        fe_or, fd_or = self.extract_features(orig)
        fe_rec, fd_dec = self.extract_features(rec)

        fe_or = fe_or[1: 4]
        fe_rec = fe_rec[1: 4]
        fd_or = fd_or[:4]
        fd_dec = fd_dec[: 4]

        feature_orig = [*fe_or, *fd_or]
        feature_rec = [*fe_rec, *fd_dec]


        diff = []

        for orig, rec in zip(feature_orig, feature_rec):
            diff.append(orig - rec)

        embeddings = diff[0]
        for layer in range(1, len(diff)):
            layer_embedding = diff[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        diff = embeddings.mean(dim=1).unsqueeze(1)
        return diff


    def calculate_pixel_level(self, orig, rec):
        diff = self.calculate_difference(orig, rec)
        anomaly_map = F.interpolate(diff, size=[self.image_size]*2, mode="nearest")
        return anomaly_map

    def calculate_image_level_anomaly_score(self, folder_normal, folder_mask, folder_rec):
        orig, rec, mask, anomaly_map, names = self.get_predictions(folder_normal, folder_mask, folder_rec)


        pixel_level_normalized = [(pl - pl.min()) / (pl.max() - pl.min()) for pl in anomaly_map]

        pixel_level = anomaly_map
        pixel_level_normalized = torch.stack(pixel_level_normalized, dim=0)

        image_level = torch.max(pixel_level.view(pixel_level.size(0), -1), dim=-1)[0]
        # image_level = 1 - structural_similarity_index_measure(orig, rec, reduction=None)
        names = [f"{round(score.item(), 3)}-{name}" for score, name in zip(image_level, names)]
        return [orig, rec, mask, pixel_level_normalized, names]

    def get_predictions(self, folder_normal, folder_mask, folder_rec):
        torch.set_grad_enabled(False)
        dl = data.DataLoader(DiffusionDataset(folder_normal, folder_mask, folder_rec, self.image_size, self.channels), batch_size=self.batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=8)

        mask_lst = []
        orig_lst = []
        rec_lst = []
        pred_lst = []

        names = []
        for idx, batch in tqdm.tqdm(enumerate(dl)):
            orig_lst.append(batch["image"])
            rec_lst.append(batch["reconstruction"])
            mask_lst.append(batch["mask"])
            pred_lst.append(self.calculate_pixel_level(self.diffusion_model.normalize(batch["image"]).cuda(), self.diffusion_model.normalize(batch["reconstruction"]).cuda()))

            names.extend([path.split(os.sep)[-1] for path in batch["path"]])
        return [torch.cat(orig_lst, dim=0), torch.cat(rec_lst, dim=0), torch.cat(mask_lst, dim=0), torch.cat(pred_lst, dim=0), names]


if __name__ == "__main__":
    test_folder_images = '/mnt/data/psemchyshyn/mvtec-diffusion/test_data'
    test_folder_mask = '/mnt/data/psemchyshyn/mvtec-diffusion/test_data_masks'
    test_folder_rec = '/mnt/data/psemchyshyn/diffusion/reconstruction100-1-threshold-10/test'
    model_path = "/mnt/data/psemchyshyn/checkpoints/updated-diffusion-mvtec-l1-self-condition/checkpoints_256/last.ckpt"

    out_path = "/mnt/data/psemchyshyn/diffusion-info/diffusion-l1-self_condition/reconstruction/test_data/rec/features"

    diffusion_model = LitModel.load_from_checkpoint(model_path).model.cuda()
    reconstructor = Predictor(diffusion_model)

    os.makedirs(out_path, exist_ok=True)

    # recs = reconstructor.reconstruct(input_path)
    recs = reconstructor.calculate_image_level_anomaly_score(test_folder_images, test_folder_mask, test_folder_rec)

    names = recs[-1]
    recs = recs[:-1]
    save_grids(out_path, *recs, names=names, grid_shape=(2, 2))

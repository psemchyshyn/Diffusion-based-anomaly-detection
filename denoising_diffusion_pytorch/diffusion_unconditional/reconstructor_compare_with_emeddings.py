import os
import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchmetrics.functional import structural_similarity_index_measure
from denoising_diffusion_pytorch.diffusion_unconditional.lightning import LitModel
from denoising_diffusion_pytorch.diffusion_unconditional.dataset import DiffusionDataset


class Reconstructor:
    def __init__(self, path_to_lit_model, normal_folder, threshold_file_path, threshold_limit=0.975, batch_size=None):
        lit = LitModel.load_from_checkpoint(path_to_lit_model).cuda()
        self.normal_folder = normal_folder
        self.ema_model = lit.ema_model
        self.image_size = self.ema_model.image_size
        self.channels = self.ema_model.channels
        self.threshold_file_path = threshold_file_path
        self.threshold_limit = threshold_limit
        self.batch_size = lit.batch_size if not batch_size else batch_size
        self.ema_model.eval()

    def calculate_difference(self, image, reconstruction, kernel_size=3):
        size = image.shape[2]
        image_pil = TF.to_pil_image(image)
        reconstruction_pil = TF.to_pil_image(reconstruction)
        to_tensor = transforms.ToTensor()

        errs = []
        for i in (1, 1/2, 1/4, 1/8):
            cur_scale_img = image_pil.resize((int(size*i), )*2)
            cur_scale_rec = reconstruction_pil.resize((int(size*i), )*2)
            err = torch.mean((to_tensor(cur_scale_img) - to_tensor(cur_scale_rec))**2, dim=0)
            errs.append(to_tensor(TF.to_pil_image(err).resize((size, )*2)))

        map = torch.mean(torch.vstack(errs), dim=0).unsqueeze(0)
        map = torch.nn.functional.avg_pool2d(map, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        return map

    def calculate_image_level_anomaly_score(self, folder_to_score, **kwargs):
        orig, noisy, rec, masks, names = self.reconstruct(folder_to_score, **kwargs)
        pixel_level = [self.calculate_difference(orig, rec) for orig, rec in zip(orig, rec)]
        pixel_level_normalized = [(pl - pl.min()) / (pl.max() - pl.min()) for pl in pixel_level]

        pixel_level = torch.stack(pixel_level, dim=0)
        pixel_level_normalized = torch.stack(pixel_level_normalized, dim=0)

        image_level = torch.max(pixel_level.view(pixel_level.size(0), -1), dim=-1)[0]

        # image_level = 1 - structural_similarity_index_measure(orig, rec, reduction=None)
        names = [f"{score.item()}-{name}" for score, name in zip(image_level, names)]
        return [orig, noisy, rec, masks, pixel_level_normalized, names]

    def calculate_pixel_level_anomaly_score(self, folder_to_score, **kwargs):
        ret = self.calculate_image_level_anomaly_score(folder_to_score, **kwargs)
        return ret[-2:]

    def reconstruct_images(self, folder, noise_amount=30, self_condition_steps=3, noise_from=300, noise_to=350, stop_self_condition_t=10):
        print(f"Reconstructing images: {folder}. Total {len(os.listdir(folder))} images")
        torch.set_grad_enabled(False)
        dl = data.DataLoader(DiffusionDataset(folder, self.image_size, self.channels), batch_size=self.batch_size,
                             shuffle=False,
                             pin_memory=True)
        image_or_lst = []
        image_rec_lst = []
        image_noisy_lst = []
        mask_lst = []

        names = []

        thresholds = self.calculate_threshold(self.normal_folder, noise_from, noise_to, self.threshold_file_path)
        for idx, batch in enumerate(dl):
            print(f"Processing batch {idx}.")
            images_or = batch["image"].cuda()

            batch_size, c, height, width = images_or.shape
            device = self.ema_model.betas.device

            images_noisy = self.ema_model.q_sample(images_or,
                                                   torch.full((batch_size,), noise_amount, device=device,
                                                              dtype=torch.long))

            images_rec, mask = self.ema_model.reconstruct(images_or, noise_amount, noise_from, noise_to, self_condition_steps, stop_self_condition_t, thresholds, self.threshold_limit)

            image_or_lst.append(images_or)
            image_noisy_lst.append(images_noisy)
            image_rec_lst.append(images_rec)
            mask_lst.append(mask)
            names.extend([path.split(os.sep)[-1] for path in batch["path"]])

        return [*list(map(lambda x: (torch.cat(x, dim=0)),
                          [image_or_lst, image_noisy_lst, image_rec_lst, mask_lst])), names]

    def calculate_threshold(self, folder, noise_from, noise_to, threshold_file_path):
        print(f"Reconstructing threshold: {folder}. Total {len(os.listdir(folder))} images. Noise from {noise_from} to {noise_to}")
        if os.path.exists(threshold_file_path):
            print(f"Loading threshold differences: {threshold_file_path}")
            return torch.load(threshold_file_path)

        dl = data.DataLoader(DiffusionDataset(folder, self.image_size, self.channels), batch_size=self.batch_size,
                             shuffle=False,
                             pin_memory=True)

        diffs = []
        for idx, batch in enumerate(dl):
            images_or = batch["image"].cuda()
            diff_tensor = self.ema_model.calculate_noise_difference(images_or, noise_from, noise_to)
            diffs.append(diff_tensor)

        diffs = torch.quantile(torch.cat(diffs, dim=1), self.threshold_limit, dim=1)
        print(f"Saving threshold differences, {diffs.shape} tensor to {threshold_file_path}")
        torch.save(diffs, threshold_file_path)
        return diffs

    def reconstruct(self, folder, **kwargs):
            return self.reconstruct_images(folder,
                                         kwargs["noise"],
                                         kwargs["self_condition_steps"],
                                         kwargs["noise_from_threshold"],
                                         kwargs["noise_to_threshold"],
                                         kwargs["stop_self_condition_t"])

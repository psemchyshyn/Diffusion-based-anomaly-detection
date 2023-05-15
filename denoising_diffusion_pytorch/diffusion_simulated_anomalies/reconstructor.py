import os

import torch
from torch.utils import data
import tqdm
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from denoising_diffusion_pytorch.diffusion_simulated_anomalies.lightning import LitModel
from denoising_diffusion_pytorch.diffusion_simulated_anomalies.dataset import DiffusionDataset


class Reconstructor:
    def __init__(self, path_to_lit_model, normal_folder, batch_size=None):
        lit = LitModel.load_from_checkpoint(path_to_lit_model).cuda()
        self.normal_folder = normal_folder
        self.ema_model = lit.ema_model
        self.image_size = self.ema_model.image_size
        self.channels = self.ema_model.channels
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

    def calculate_threslhold(self, folder, threshold_file_path, **kwargs):
        if os.path.exists(threshold_file_path):
            return torch.load(threshold_file_path)

        ret_val = self.reconstruct(folder, **kwargs)
        diff = ret_val[-2]

        threshold = torch.mean(diff, dim=0)
        torch.save(threshold, threshold_file_path)
        return threshold

    def calculate_image_level_anomaly_score(self, folder_to_score, threshold_file_path, **kwargs):
        # threshold = self.calculate_threslhold(self.normal_folder, threshold_file_path, **kwargs)
        orig, noisy, rec, diff, names = self.reconstruct(folder_to_score, **kwargs)
        pixel_level = diff
        pixel_level_normalized = (pixel_level - pixel_level.min()) / (pixel_level.max() - pixel_level.min())
        image_level, _ = torch.max(pixel_level.view(pixel_level.size(0), -1), dim=-1)

        # threshold -= threshold.min()
        # threshold /= threshold.max()
        # threshold = threshold.unsqueeze(dim=0).repeat(pixel_level.shape[0], 1, 1)
        # pixel_level -= pixel_level.min()
        # pixel_level /= pixel_level.max()
        # diff = transforms.Normalize(mean=[-0.5 / 0.5], std=[1 / 0.5])(diff)
        diff = [(d - d.min()) / (d.max() - d.min()) for d in diff]
        diff = torch.stack(diff, dim=0).unsqueeze(1)
        print(diff.shape)
        print(diff)
        names = [f"{score.item()}-{name}" for score, name in zip(image_level, names)]
        return [orig, noisy, rec, pixel_level_normalized, diff, names]

    def reconstruct_without_cropping(self, folder, noise_amount=500, self_condition_steps=1):
        dl = data.DataLoader(DiffusionDataset(folder, self.image_size, self.channels), batch_size=self.batch_size,
                             shuffle=False,
                             pin_memory=True)
        image_or_lst = []
        image_rec_lst = []
        image_noisy_lst = []
        image_diff_lst = []
        names = []

        for idx, batch in enumerate(dl):
            images_or = batch["image"].cuda()

            batch_size, c, height, width = images_or.shape
            device = self.ema_model.betas.device
            noise = torch.randn_like(images_or)

            images_noisy = self.ema_model.q_sample(images_or,
                                                   torch.full((batch_size,), noise_amount, device=device,
                                                              dtype=torch.long), noise=noise)

            images_rec = self.ema_model.reconstruct_simple(images_noisy, noise_amount, torch.zeros((batch_size, ), device=device).long(), self_condition_steps)

            image_or_lst.append(images_or)
            image_noisy_lst.append(images_noisy)
            image_rec_lst.append(images_rec)
            image_diff_lst.append(torch.vstack([self.calculate_difference(el1, el2) for el1, el2 in zip(images_or, images_rec)]))
            names.extend([path.split(os.sep)[-1] for path in batch["path"]])

        return [*list(map(lambda x: (torch.cat(x, dim=0)),
                          [image_or_lst, image_noisy_lst, image_rec_lst, image_diff_lst])), names]


    def reconstruct(self, folder, **kwargs):
            return self.reconstruct_without_cropping(folder, kwargs["noise"], kwargs["self_condition_steps"])

import os

import torch
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from denoising_diffusion_pytorch.diffusion_unconditional.lightning import LitModel
from denoising_diffusion_pytorch.diffusion_unconditional.dataset import DiffusionDataset
from denoising_diffusion_pytorch.diffusion_unconditional.memory_bank import MemoryBank


class Reconstructor:
    def __init__(self, path_to_lit_model, normal_folder, batch_size=None):
        lit = LitModel.load_from_checkpoint(path_to_lit_model).cuda()
        self.normal_folder = normal_folder
        self.storage = MemoryBank(normal_folder)
        self.ema_model = lit.ema_model
        self.image_size = self.ema_model.image_size
        self.channels = self.ema_model.channels
        self.batch_size = lit.batch_size if not batch_size else batch_size

        self.ema_model.eval()
        self.storage.compute_memory_bank()

    def calculate_difference(self, image, reconstruction, kernel_size=5):
        size = image.shape[2]
        image_pil = TF.to_pil_image(image)
        reconstruction_pil = TF.to_pil_image(reconstruction)
        to_tensor = transforms.ToTensor()

        errs = []
        for i in (1, 1 / 2, 1 / 4, 1 / 8):
            cur_scale_img = image_pil.resize((int(size * i),) * 2)
            cur_scale_rec = reconstruction_pil.resize((int(size * i),) * 2)
            err = torch.mean((to_tensor(cur_scale_img) - to_tensor(cur_scale_rec)) ** 2, dim=0)
            errs.append(to_tensor(TF.to_pil_image(err).resize((size,) * 2)))

        # print(errs)
        map = torch.mean(torch.vstack(errs), dim=0).unsqueeze(0)
        # print(map)
        map = torch.nn.functional.avg_pool2d(map, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        map -= map.min()
        map /= map.max()
        return map

    def reconstruct(self, folder, noise_amount=500, self_condition_steps=1):
        torch.set_grad_enabled(False)
        dl = data.DataLoader(DiffusionDataset(folder, self.image_size, self.channels), batch_size=self.batch_size,
                             shuffle=False,
                             pin_memory=True)
        image_or_lst = []
        image_rec_lst = []
        image_noisy_lst = []
        image_diff_lst = []

        images_closest_lst = []
        images_closest_noisy_lst = []
        images_closest_rec_lst = []
        images_closest_diff_lst = []

        names = []

        for idx, batch in enumerate(dl):
            images_or = batch["image"].cuda()
            embeddings, closest_names, _ = self.storage.select([Image.open(path) for path in batch["path"]])
            images_closest = torch.stack(
                [transforms.ToTensor()(Image.open(path).convert("RGB").resize((self.ema_model.image_size,) * 2)) for
                 path in
                 closest_names], dim=0).cuda()

            batch_size, c, height, width = images_or.shape
            device = self.ema_model.betas.device
            noise = torch.randn_like(images_or)

            images_noisy = self.ema_model.q_sample(images_or,
                                                   torch.full((batch_size,), noise_amount, device=device,
                                                              dtype=torch.long), noise=noise)

            images_closest_noisy = self.ema_model.q_sample(images_closest,
                                                           torch.full((batch_size,), noise_amount, device=device,
                                                                      dtype=torch.long), noise=noise)

            images_rec = self.ema_model.reconstruct_simple(images_noisy, noise_amount, self_condition_steps)
            # images_closest_rec = self.ema_model.reconstruct_simple(images_closest_noisy, noise_amount,
            #                                                        self_condition_steps)

            image_or_lst.append(images_or)
            image_noisy_lst.append(images_noisy)
            image_rec_lst.append(images_rec)
            images_closest_lst.append(images_closest)
            images_closest_noisy_lst.append(images_closest_noisy)
            images_closest_rec_lst.append(images_rec)

            image_diff_lst.append(
                torch.vstack([self.calculate_difference(el1, el2) for el1, el2 in zip(images_or, images_rec)]))
            images_closest_diff_lst.append(torch.vstack(
                [self.calculate_difference(el1, el2) for el1, el2 in zip(images_closest, images_rec)]))
            names.extend([path.split(os.sep)[-1] for path in batch["path"]])

        return [*list(map(lambda x: (torch.cat(x, dim=0)),
                          [image_or_lst, image_noisy_lst, image_rec_lst, image_diff_lst, images_closest_lst,
                           images_closest_noisy_lst, images_closest_rec_lst, images_closest_diff_lst])), names]


    def calculate_image_level_anomaly_score(self, folder_to_score, **kwargs):
        orig, noisy, rec, diff, closest, closest_noisy, closest_rec, closest_diff, names = self.reconstruct(
            folder_to_score, kwargs["noise"], kwargs["self_condition_steps"])

        closest_features, _, distance_orig_closest = self.storage.select([TF.to_pil_image(i) for i in orig], 6)


        closest_to_rec_features, _, distance_rec_closest = self.storage.select([TF.to_pil_image(i) for i in rec], 6)


        image_level = distance_orig_closest - distance_rec_closest
        names = [f"{score.item()}-{name}" for score, name in zip(image_level, names)]
        return [orig, noisy, rec, diff, closest, closest_noisy, closest_rec, closest_diff, names]

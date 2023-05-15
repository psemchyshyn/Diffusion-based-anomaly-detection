import os

import torch
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from denoising_diffusion_pytorch.lightning import LitModel
from denoising_diffusion_pytorch.dataset import DiffusionDataset
from anomaly_detection.fill_patch_strategies import triangle_reflection, outpatch_mean


class Reconstructor:
    def __init__(self, path_to_lit_model, batch_size=None):
        lit = LitModel.load_from_checkpoint(path_to_lit_model).cuda()
        self.storage = lit.storage
        self.model = lit.model
        self.ema_model = lit.ema_model
        self.image_size = self.model.image_size
        self.channels = self.model.channels
        self.batch_size = lit.batch_size if not batch_size else batch_size

        self.storage.compute_memory_bank()

    def calculate_difference(self, image, reconstruction, kernel_size=4):
        size = image.shape[2]
        image_pil = TF.to_pil_image(image)
        reconstruction_pil = TF.to_pil_image(reconstruction)
        to_tensor = transforms.ToTensor()

        errs = []
        for i in (1, 1/2, 1/4, 1/8):
            cur_scale_img = image_pil.resize((int(size*i), )*2)
            cur_scale_rec = reconstruction_pil.resize((int(size*i), )*2)
            err = torch.mean((to_tensor(cur_scale_img) - to_tensor(cur_scale_rec))**2, dim=0)
            # err = self.ema_model.normalize(err)

            errs.append(to_tensor(TF.to_pil_image(err).resize((size, )*2)))

        # print(errs)
        map = torch.mean(torch.vstack(errs), dim=0).unsqueeze(0)
        # print(map)
        map = torch.nn.functional.avg_pool2d(map, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        map -= map.min()
        map /= map.max()
        return map

    def calculate_threslhold(self, folder, type, threshold_file_path, **kwargs):
        if os.path.exists(threshold_file_path):
            return torch.load(threshold_file_path)

        ret_val = self.reconstruct(folder, type, **kwargs)
        diff = ret_val[-2]

        threshold = torch.mean(diff, dim=0)
        torch.save(threshold, threshold_file_path)
        return threshold

    def calculate_image_level_anomaly_score(self, folder_to_score, normal_folder, threshold_file_path, type, **kwargs):
        threshold = self.calculate_threslhold(normal_folder, type, threshold_file_path, **kwargs)

        orig, noisy, closest, rec, diff, names = self.reconstruct(folder_to_score, type, **kwargs)
        pixel_level = diff - threshold
        image_level, _ = torch.max(pixel_level.view(pixel_level.size(0), -1), dim=-1)
        threshold -= threshold.min()
        threshold /= threshold.max()
        threshold = threshold.unsqueeze(dim=0).repeat(pixel_level.shape[0], 1, 1)
        pixel_level -= pixel_level.min()
        pixel_level /= pixel_level.max()
        names = [f"{score.item()}-{name}" for score, name in zip(image_level, names)]
        return [orig, noisy, closest, rec, diff, pixel_level, threshold, names]


    # def reconstruct_without_cropping(self, folder, noise_amount=500):
    #     torch.set_grad_enabled(False)
    #     dl = data.DataLoader(DiffusionDataset(folder, self.image_size, self.channels), batch_size=self.batch_size, shuffle=False,
    #                          pin_memory=True)
    #     image_or_lst = []
    #     image_rec_lst = []
    #     image_noisy_lst = []
    #     image_diff_lst = []
    #
    #     images_closest_lst = []
    #     images_closest_noisy_lst = []
    #     images_rec_closest_lst = []
    #     images_closest_diff_lst = []
    #     names = []
    #
    #     for idx, batch in enumerate(dl):
    #         images_or = batch["image"].cuda()
    #         embeddings, closest_names = self.storage.select([Image.open(path) for path in batch["path"]])
    #
    #
    #         embeddings = embeddings.cuda()
    #         idx = torch.randperm(embeddings.shape[0])
    #         embeddings = embeddings[idx]
    #         closest_names = closest_names[idx]
    #
    #
    #         images_closest = torch.stack([transforms.ToTensor()(Image.open(path).resize((self.ema_model.image_size,)*2).convert("RGB")) for path in closest_names], dim=0).cuda()
    #
    #         batch_size, c, height, width = images_or.shape
    #         self.ema_model.eval()
    #         device = self.ema_model.betas.device
    #         images_noisy = self.ema_model.q_sample(images_or,
    #                                                torch.full((batch_size,), noise_amount, device=device,
    #                                                           dtype=torch.long))
    #
    #         images_closest_noisy = self.ema_model.q_sample(images_closest,
    #                                                torch.full((batch_size,), noise_amount, device=device,
    #                                                           dtype=torch.long))
    #
    #         # images_rec = self.ema_model.reconstruct(noisy_list[-1], embeddings, noise_amount)
    #         images_rec = self.ema_model.reconstruct(images_noisy, embeddings, noise_amount)
    #         images_rec_closest = self.ema_model.reconstruct(images_closest_noisy, embeddings, noise_amount)
    #
    #         image_rec_lst.append(images_rec)
    #         image_noisy_lst.append(images_noisy)
    #         image_or_lst.append(images_or)
    #
    #         images_closest_lst.append(images_closest)
    #         images_closest_noisy_lst.append(images_closest_noisy)
    #         images_rec_closest_lst.append(images_rec_closest)
    #
    #         image_diff_lst.append(self.ema_model.normalize(torch.abs(images_or - images_rec).mean(dim=1)))
    #         images_closest_diff_lst.append(self.ema_model.normalize(torch.abs(images_closest - images_rec_closest).mean(dim=1)))
    #
    #         names.extend([path.split(os.sep)[-1] for path in batch["path"]])
    #
    #
    #     return [*list(map(lambda x: (torch.cat(x, dim=0)), [image_or_lst, image_noisy_lst, image_rec_lst, images_closest_lst, images_closest_noisy_lst, images_rec_closest_lst, image_diff_lst, images_closest_diff_lst])), names]

    def reconstruct_without_cropping(self, folder, noise_amount=500, self_condition_steps=1):
        torch.set_grad_enabled(False)
        dl = data.DataLoader(DiffusionDataset(folder, self.image_size, self.channels), batch_size=self.batch_size,
                             shuffle=False,
                             pin_memory=True)
        image_or_lst = []
        image_rec_lst = []
        image_noisy_lst = []
        image_diff_lst = []

        images_closest_lst = []
        names = []

        for idx, batch in enumerate(dl):
            images_or = batch["image"].cuda()
            embeddings, closest_names = self.storage.select([Image.open(path) for path in batch["path"]])
            images_closest = torch.stack(
                [transforms.ToTensor()(Image.open(path).convert("RGB").resize((self.ema_model.image_size,) * 2)) for path in
                 closest_names], dim=0).cuda()

            embeddings = embeddings.cuda()
            # idx = torch.randperm(embeddings.shape[0])
            # embeddings = embeddings[idx]
            # images_closest = images_closest[idx]


            batch_size, c, height, width = images_or.shape
            self.ema_model.eval()
            device = self.ema_model.betas.device



            images_noisy = self.ema_model.q_sample(images_or,
                                                   torch.full((batch_size,), noise_amount, device=device,
                                                              dtype=torch.long))

            # images_closest_noisy = self.ema_model.q_sample(images_closest,
            #                                                torch.full((batch_size,), noise_amount, device=device,
            #                                                           dtype=torch.long))

            images_rec = self.ema_model.reconstruct(images_or, embeddings, noise_amount, self_condition_steps)



            image_rec_lst.append(images_rec)
            image_noisy_lst.append(images_noisy)
            image_or_lst.append(images_or)
            images_closest_lst.append(images_closest)

            image_diff_lst.append(torch.vstack([self.calculate_difference(el1, el2) for el1, el2 in zip(images_or, images_rec)]))
            names.extend([path.split(os.sep)[-1] for path in batch["path"]])

        return [*list(map(lambda x: (torch.cat(x, dim=0)),
                          [image_or_lst, image_noisy_lst, images_closest_lst, image_rec_lst, image_diff_lst])), names]


    def reconstruct_cropping(self, folder, crop_mult=4, fill_patch_val=0, noise_amount=500):
        dl = data.DataLoader(DiffusionDataset(folder, self.image_size, self.channels), batch_size = self.batch_size, shuffle=False, pin_memory=True)
        image_or_lst = []
        image_rec_lst = []
        image_cropped_lst = []
        image_diff_lst = []
        for idx, batch in enumerate(dl):
            images_or = batch.cuda()

            batch_size, c, height, width = images_or.shape
            images_cropped = torch.clone(images_or)

            if fill_patch_val == "mean":
                val = outpatch_mean(torch.clone(images_cropped), crop_mult)
            elif fill_patch_val == "reflection":
                val = triangle_reflection(torch.clone(images_cropped), crop_mult)[:, :, height // crop_mult: -height // crop_mult, width // crop_mult: -width // crop_mult]
            else:
                val = fill_patch_val*2 - 1

            images_cropped[:, :, (height // crop_mult): (-height // crop_mult) + 1, (width // crop_mult): (-width // crop_mult) + 1] = val

            self.ema_model.eval()
            device = self.ema_model.betas.device
            images_noisy = self.ema_model.q_sample(images_cropped, torch.full((batch_size, ), noise_amount, device=device, dtype=torch.long))
            images_rec = self.ema_model.reconstruct(images_noisy, noise_amount)

            image_rec_lst.append(images_rec)
            image_cropped_lst.append(images_cropped)
            image_or_lst.append(images_or)
            image_diff_lst.append((torch.abs(images_or - images_rec)))


        return list(map(lambda x: (torch.cat(x, dim=0) + 1)*0.5, [image_or_lst, image_cropped_lst, image_rec_lst, image_diff_lst]))

    def reconstruct_resampling(self, folder, crop_mult=4, fill_patch_val=0, noise_amount=500, resampling=10):
        dl = data.DataLoader(DiffusionDataset(folder, self.image_size, self.channels), batch_size = self.batch_size, shuffle=False, pin_memory=True)
        image_or_lst = []
        image_rec_lst = []
        image_cropped_lst = []
        image_diff_lst = []

        for idx, batch in enumerate(dl):
            images_or = batch.cuda()
            batch_size, c, height, width = images_or.shape

            images_cropped = torch.clone(images_or)

            if fill_patch_val == "mean":
                val = outpatch_mean(torch.clone(images_cropped), crop_mult)
            elif fill_patch_val == "reflection":
                val = triangle_reflection(torch.clone(images_cropped), crop_mult)[:, :, height // crop_mult: -height // crop_mult, width // crop_mult: -width // crop_mult]
            else:
                val = fill_patch_val*2 - 1

            images_cropped[:, :, (height // crop_mult): (-height // crop_mult) + 1, (width // crop_mult): (-width // crop_mult) + 1] = val

            self.ema_model.eval()
            device = self.ema_model.betas.device

            mask = torch.zeros_like(images_cropped)
            mask[:, :, height // crop_mult: -height // crop_mult, width // crop_mult: -width // crop_mult] = 1
            noise = torch.randn_like(images_cropped)

            image_t = self.ema_model.q_sample(images_cropped, torch.full((batch_size, ), noise_amount, device=device, dtype=torch.long), noise)
            for i in reversed(range(1, noise_amount + 1)):
                t_prev = torch.full((batch_size,), i - 1, device=device, dtype=torch.long)
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                noise = torch.randn_like(images_cropped)
                image_before_t_q = self.ema_model.q_sample(images_cropped, t_prev, noise)
                for j in range(resampling):
                    image_before_t_p = self.ema_model.p_sample(image_t, t)
                    image_t = mask*image_before_t_p + (1 - mask)*image_before_t_q
                    if j != resampling - 1:
                        image_t = self.ema_model.q_sample_t_from_prev(image_t, t, noise)


            images_rec = image_t
            image_rec_lst.append(images_rec)
            image_cropped_lst.append(images_cropped)
            image_or_lst.append(images_or)
            image_diff_lst.append((torch.abs(images_or - images_rec)))

        return list(map(lambda x: (torch.cat(x, dim=0) + 1)*0.5, [image_or_lst, image_cropped_lst, image_rec_lst, image_diff_lst]))

    def reconstruct_border_aware_resampling(self, folder, crop_mult=4, fill_patch_val=0, noise_amount=500, resampling=10):
        dl = data.DataLoader(DiffusionDataset(folder, self.image_size), batch_size=self.batch_size,
                             shuffle=False, pin_memory=True)
        image_or_lst = []
        image_rec_lst = []
        image_cropped_lst = []
        image_diff_lst = []
        for idx, batch in enumerate(dl):
            images_or = batch.cuda()
            batch_size, c, height, width = images_or.shape

            images_cropped = torch.clone(images_or)

            if fill_patch_val == "mean":
                val = outpatch_mean(torch.clone(images_cropped), crop_mult)
            elif fill_patch_val == "reflection":
                val = triangle_reflection(torch.clone(images_cropped), crop_mult)[:, :,
                      height // crop_mult: -height // crop_mult, width // crop_mult: -width // crop_mult]
            else:
                val = fill_patch_val * 2 - 1

            images_cropped[:, :, height // crop_mult: -height // crop_mult,
            width // crop_mult: -width // crop_mult] = val

            self.ema_model.eval()
            device = self.ema_model.betas.device

            mask = torch.zeros_like(images_cropped)
            mask[:, :, height // crop_mult: -height // crop_mult, width // crop_mult: -width // crop_mult] = 1
            noise = torch.randn_like(images_cropped)

            image_t = self.ema_model.q_sample(images_cropped,
                                              torch.full((batch_size,), noise_amount, device=device, dtype=torch.long),
                                              noise)
            real_mask_coef = 0.99
            for i in reversed(range(1, noise_amount + 1)):
                t_prev = torch.full((batch_size,), i - 1, device=device, dtype=torch.long)
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                noise = torch.randn_like(images_cropped)
                real_mask_coef = max(0, real_mask_coef - 0.01)
                image_before_t_q = self.ema_model.q_sample(images_cropped, t_prev, noise)
                for j in range(resampling):
                    image_before_t_p = self.ema_model.p_sample(image_t, t)
                    border = real_mask_coef*image_before_t_q + (1 - real_mask_coef)*image_before_t_p
                    image_t = mask * image_before_t_p + (1 - mask) * border
                    if j != resampling - 1:
                        image_t = self.ema_model.q_sample_t_from_prev(image_t, t, noise)


            images_rec = image_t
            image_rec_lst.append(images_rec)
            image_cropped_lst.append(images_cropped)
            image_or_lst.append(images_or)
            image_diff_lst.append((torch.abs(images_or - images_rec)))

        return list(map(lambda x: (torch.cat(x, dim=0) + 1)*0.5, [image_or_lst, image_cropped_lst, image_rec_lst, image_diff_lst]))

    def reconstruct(self, folder, type, **kwargs):
        if type == "standard":
            return self.reconstruct_cropping(folder, kwargs["patch_ratio"], kwargs["fill_value"], kwargs["noise"])
        elif type == "without_crop":
            return self.reconstruct_without_cropping(folder, kwargs["noise"], kwargs["self_condition_steps"])
        elif type == "resampling":
            return self.reconstruct_resampling(folder, kwargs["patch_ratio"], kwargs["fill_value"], kwargs["noise"], kwargs["resampling"])
        elif type == "border_aware_resampling":
            return self.reconstruct_border_aware_resampling(folder, kwargs["patch_ratio"], kwargs["fill_value"], kwargs["noise"], kwargs["resampling"])

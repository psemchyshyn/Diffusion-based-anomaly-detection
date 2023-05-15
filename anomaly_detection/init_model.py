import torch
import yaml
from torch.utils import data
from denoising_diffusion_pytorch import *
from anomaly_detection.fill_patch_strategies import triangle_reflection, outpatch_mean

torch.cuda.set_device(0)

class TrainerReconstruction(Trainer):
    def __init__(self, *args, **kwargs):
        super(TrainerReconstruction, self).__init__(*args, **kwargs)

    def reconstruct_without_cropping(self, folder, noise_amount=500):
        dl = data.DataLoader(Dataset(folder, self.image_size, do_transform=False), batch_size=self.batch_size, shuffle=False,
                             pin_memory=True)
        image_or_lst = []
        image_rec_lst = []
        image_noisy_lst = []
        image_diff_lst = []

        for idx, batch in enumerate(dl):
            images_or = batch.cuda()

            batch_size, c, height, width = images_or.shape

            self.ema_model.eval()
            device = self.ema_model.betas.device
            images_noisy = self.ema_model.q_sample(images_or,
                                                   torch.full((batch_size,), noise_amount, device=device,
                                                              dtype=torch.long))
            images_rec = self.ema_model.reconstruct(images_noisy, noise_amount)

            image_rec_lst.append(images_rec)
            image_noisy_lst.append(images_noisy)
            image_or_lst.append(images_or)
            image_diff_lst.append((torch.abs(images_or - images_rec)))


        return list(map(lambda x: (torch.cat(x, dim=0) + 1) * 0.5,
                        [image_or_lst, image_noisy_lst, image_rec_lst, image_diff_lst]))

    def reconstruct_cropping(self, folder, crop_mult=4, fill_patch_val=0, noise_amount=500):
        dl = data.DataLoader(Dataset(folder, self.image_size, do_transform=False), batch_size = self.batch_size, shuffle=False, pin_memory=True)
        image_or_lst = []
        image_rec_lst = []
        image_noisy_lst = []
        image_cropped_lst = []
        image_diff_lst = []
        for idx, batch in enumerate(dl):
            images_or = batch.cuda(0)

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
            image_noisy_lst.append(images_noisy)
            image_cropped_lst.append(images_cropped)
            image_or_lst.append(images_or)
            image_diff_lst.append((torch.abs(images_or - images_rec)))


        return list(map(lambda x: (torch.cat(x, dim=0) + 1)*0.5, [image_or_lst, image_cropped_lst, image_rec_lst, image_diff_lst]))

    def reconstruct_resampling(self, folder, crop_mult=4, fill_patch_val=0, noise_amount=500, resampling=10):
        dl = data.DataLoader(Dataset(folder, self.image_size, do_transform=False), batch_size = self.batch_size, shuffle=False, pin_memory=True)
        image_or_lst = []
        image_rec_lst = []
        image_cropped_lst = []
        image_diff_lst = []

        for idx, batch in enumerate(dl):
            images_or = batch.cuda(0)
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
            image_diff_lst.append(torch.abs(images_or - images_rec))

        return list(map(lambda x: (torch.cat(x, dim=0) + 1)*0.5, [image_or_lst, image_cropped_lst, image_rec_lst, image_diff_lst]))

    def reconstruct_border_aware_resampling(self, folder, crop_mult=4, fill_patch_val=0, noise_amount=500, resampling=10):
        dl = data.DataLoader(Dataset(folder, self.image_size, do_transform=False), batch_size=self.batch_size,
                             shuffle=False, pin_memory=True)
        image_or_lst = []
        image_rec_lst = []
        image_cropped_lst = []
        image_diff_lst = []
        for idx, batch in enumerate(dl):
            images_or = batch.cuda(self.ema_model.betas.device)
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

        return list(map(lambda x: (torch.cat(x, dim=0) + 1) * 0.5, [image_or_lst, image_cropped_lst, image_rec_lst, image_diff_lst]))

    def reconstruct(self, folder, type, **kwargs):
        if type == "standard":
            return self.reconstruct_cropping(folder, kwargs["patch_ratio"], kwargs["fill_value"], kwargs["noise"])
        elif type == "without_crop":
            return self.reconstruct_without_cropping(folder, kwargs["noise"])
        elif type == "resampling":
            return self.reconstruct_resampling(folder, kwargs["patch_ratio"], kwargs["fill_value"], kwargs["noise"], kwargs["resampling"])
        elif type == "border_aware_resampling":
            return self.reconstruct_border_aware_resampling(folder, kwargs["patch_ratio"], kwargs["fill_value"], kwargs["noise"], kwargs["resampling"])


with open("anomaly_detection/config.yaml", 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)


def create_model(config):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=config["model"]["params"]["channels"]
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=config["model"]["params"]["image_size"],
        noise_type="gaussian",
        timesteps=1000,  # number of steps
        channels=config["model"]["params"]["channels"],
        loss_type='l1'  # L1 or L2
    ).cuda()

    trainer = TrainerReconstruction(
        diffusion,
        config["model"]["data"]["train_path"],
        config["model"]["data"]["val_path"],
        image_size=config["model"]["params"]["image_size"],
        train_batch_size=config["model"]["data"]["batch_size"],
        train_lr=2e-5,
        epochs=3000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        fp16=False,  # turn on mixed precision training with apex
        results_folder=config["model"]["data"]["results_folder"],
        log_dir='eruns'
    )

    return trainer

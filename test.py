import os
from PIL import Image


import os
import torch
from utils import save_grids
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from classification.dataset import DiffusionDataset
import tqdm


class Predictor:
    def __init__(self, image_size=256, channels=3, batch_size=64):
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size


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

    def calculate_image_level_anomaly_score(self, folder_normal, folder_mask, folder_rec):
        orig, rec, mask, names = self.get_predictions(folder_normal, folder_mask, folder_rec)


        pixel_level = [self.calculate_difference(orig, rec) for orig, rec in zip(orig, rec)]
        pixel_level_normalized = [(pl - pl.min()) / (pl.max() - pl.min()) for pl in pixel_level]

        pixel_level = torch.stack(pixel_level, dim=0)
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

        names = []
        for idx, batch in tqdm.tqdm(enumerate(dl)):
            orig_lst.append(batch["image"])
            rec_lst.append(batch["reconstruction"])
            mask_lst.append(batch["mask"])

            names.extend([path.split(os.sep)[-1] for path in batch["path"]])
        return [torch.cat(orig_lst, dim=0), torch.cat(rec_lst, dim=0), torch.cat(mask_lst, dim=0), names]


if __name__ == "__main__":
    test_folder_images = '/mnt/data/psemchyshyn/mvtec-diffusion/test_data'
    test_folder_mask = '/mnt/data/psemchyshyn/mvtec-diffusion/test_data_masks'
    test_folder_rec = '/mnt/data/psemchyshyn/diffusion/reconstruction250-1-threshold-0/test'

    out_path = "/mnt/data/psemchyshyn/diffusion-info/diffusion-l1-self_condition/reconstruction/test_data/rec/250-1-threshold-0"
    reconstructor = Predictor()

    os.makedirs(out_path, exist_ok=True)

    # recs = reconstructor.reconstruct(input_path)
    recs = reconstructor.calculate_image_level_anomaly_score(test_folder_images, test_folder_mask, test_folder_rec)

    names = recs[-1]
    recs = recs[:-1]
    save_grids(out_path, *recs, names=names, grid_shape=(2, 2))

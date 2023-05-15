from segmentation.augmenter import Augmenter
from denoising_diffusion_pytorch.diffusion_unconditional.dataset import DiffusionDataset
from torch.utils import data
import torchvision.transforms.functional as TF
import os


if __name__ == "__main__":
    folder = "/mnt/data/psemchyshyn/mvtec-diffusion/train_data"
    dtd_path = "/mnt/data/psemchyshyn/mvtec-diffusion/dtd/images"
    image_size = 256
    channels = 3
    out_folder_image = "/mnt/data/psemchyshyn/mvtec-diffusion-subnetwork/train/all"
    out_folder_mask = "/mnt/data/psemchyshyn/mvtec-diffusion-subnetwork/train/mask"
    ds = DiffusionDataset(folder, image_size, channels)
    dl = data.DataLoader(ds, batch_size=32, shuffle=False)
    augmenter = Augmenter(dtd_path)

    for batch in dl:
        paths = list(map(lambda x: x.split(os.sep)[-1], batch["path"]))
        augmented, masks = augmenter.augment_batch(batch["image"])

        augmented = [TF.to_pil_image(image) for image in augmented]
        masks = [TF.to_pil_image(image) for image in masks]
        for i, (image, mask) in enumerate(zip(augmented, masks)):
            image.save(os.path.join(out_folder_image, paths[i]))
            mask.save(os.path.join(out_folder_mask, paths[i]))





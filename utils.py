from anomaly_detection.metrics import *
import numpy as np
import os
import torchvision.utils as utils
import torchvision.transforms.functional as TF
from PIL import Image


def image_grid(imgs, rows, cols):

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def save_grids(folder, *args, names=None, grid_shape=(2, 2)):
    os.makedirs(folder, exist_ok=True)
    names = [f"{i}.png" for i in range(len(args[0]))] if names is None else names
    for i, img_set in enumerate(zip(*args)):
        # print("Image set", img_set)
        imgs = []
        for img in img_set:
            imgs.append(TF.to_pil_image(img))

        # imgs[-1] = Image.blend(imgs[0], imgs[-1], 0.2)
        res_image = image_grid(imgs, *grid_shape)
        res_image.save(f"{folder}/{names[i]}")

# def save_grids(folder, *args, names=None):
#     os.makedirs(folder, exist_ok=True)
#     names = [f"{i}.png" for i in range(len(args[0]))] if names is None else names
#     for i, img_set in enumerate(zip(*args)):
#         # print("Image set", img_set)
#         grid = utils.make_grid(list(img_set), nrow=2)
#         utils.save_image(grid, f"{folder}/{names[i]}")

def get_score(images_or, images_rec, metric, patch_ratio, patch_only):
    if patch_only:
        _, _, height, width = images_or.shape
        images_or = images_or[:, :, height // patch_ratio: -height // patch_ratio,
                    width // patch_ratio: -width // patch_ratio]
        images_rec = images_rec[:, :, height // patch_ratio: -height // patch_ratio,
                     width // patch_ratio: -width // patch_ratio]

    scores = []
    for i in range(len(images_or)):
        scores.append(metric(images_or[i].unsqueeze(0).cpu(), images_rec[i].unsqueeze(0).cpu()).item())
    return scores

def get_reconstructed(folder, model, method, fill_value, patch_ratio, diffusion_steps, resampling_steps):
    images_or, images_cropped, *_, images_rec, images_diff = model.reconstruct(folder, method, fill_value=fill_value, patch_ratio=patch_ratio,
                                                                        noise=diffusion_steps, resampling=resampling_steps)
    return images_or, images_cropped, images_rec, images_diff

def get_reconstructed_score(folder, model, strategy, patch_ratio, fill_val, diffusion_steps, resampling_steps=None, metric=MSE(), patch_only=True, save_to_scores="scores.npy"):
    orig, cropped, rec, diff = get_reconstructed(folder, model, strategy, fill_val, patch_ratio, diffusion_steps, resampling_steps)
    scores = get_score(orig, rec, metric, patch_ratio, patch_only)
    scores = np.array(scores)
    # pos = np.array(list(map(lambda x: int(x[:-4]), os.listdir(folder))))
    pos = np.array(os.listdir(folder))
    sorted_idx = np.argsort(pos)
    scores = scores[sorted_idx]
    np.save(save_to_scores, scores)

    return orig, cropped, rec, diff

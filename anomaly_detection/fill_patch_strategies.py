import torch

def triangle_reflection(images, patch_ratio):
    *_, height, width = images.shape
    steps_width = (width - 2 * width // patch_ratio) // 2
    steps_height = (height - 2 * height // patch_ratio) // 2
    height_offset = height // patch_ratio
    width_offset = width // patch_ratio
    for i in range(1, min(steps_width, height_offset)):
        images[:, :, height_offset + i: -height_offset - i, width_offset + i] = images[:, :, height_offset + i: -height_offset - i, width_offset - i]
        images[:, :, height_offset + i: -height_offset - i, -width_offset - i] = images[:, :, height_offset + i: -height_offset - i, -width_offset + i]

    for i in range(1, min(steps_height, width_offset)):
        images[:, :, height_offset + i, width_offset + i: -width_offset - i] = images[:, :, height_offset - i, width_offset + i: -width_offset - i]
        images[:, :, -height_offset - i, width_offset + i: -width_offset - i] = images[:, :, -height_offset + i, width_offset + i: -width_offset - i]

    return images


def outpatch_mean(images, patch_ratio):
    mask = torch.zeros_like(images)
    *_, height, width = images.shape
    mask[:, :, height // patch_ratio: -height // patch_ratio, width // patch_ratio: -width // patch_ratio] = 1
    cropped = (1 - mask)*images
    mean = cropped.sum(dim=(2, 3)) / (1 - mask).sum(dim=(2, 3))
    mean = mean.unsqueeze(2)
    mean = mean.unsqueeze(3)
    mean = mean.expand(-1, -1, height - 2*int(height / patch_ratio), width - 2*int(width / patch_ratio))
    return mean

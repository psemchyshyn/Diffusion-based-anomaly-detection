import cv2
from random import choice, uniform
import glob
import numpy as np


def noise_foreground_generate(ori, mask, texture, trans_range=(0.15, 1)):
    texture_image_path = choice(glob.glob(f"{texture}/**/*"))
    random_num = uniform(1, 10)
    if random_num <= 5:
        i_n = cv2.imread(texture_image_path)
        i_n = cv2.resize(i_n, ori.shape[:2][::-1])
    else:
        (h, w) = ori.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), uniform(30, 180), 1.0)
        rotated = cv2.warpAffine(ori, M, (w, h))
        i_n = rotated

    mask = mask.astype(np.uint8)
    i_n = i_n.astype(np.uint8)
    ori = ori.astype(np.uint8)

    factor = uniform(trans_range[0], trans_range[1])
    i_n_r_2 = cv2.bitwise_and(ori, ori, mask=mask)
    i_n_r_1 = cv2.bitwise_and(i_n, i_n, mask=mask)
    i_n_r = factor * i_n_r_1 + (1 - factor) * i_n_r_2
    return i_n_r
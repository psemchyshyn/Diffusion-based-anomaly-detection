import cv2
import numpy as np



def simulated_generate(mask, ori, noisy):
    mask = np.where(mask == 255, 0, 255).astype(np.uint8)
    i_r = cv2.bitwise_and(ori, ori, mask=mask)
    res = i_r + noisy
    return res
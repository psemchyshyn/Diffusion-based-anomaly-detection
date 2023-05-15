import cv2
from noise import pnoise2
import numpy as np
from perlin_noise import PerlinNoise


class MaskImage:
    def __init__(self, path, patch_ratio=6, octaves=2, persistence=0.5, lacunarity=2.0):
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

        self.path = path
        self.img = cv2.imread(self.path)
        self.shape = self.img.shape[0:2]
        self.patch_ratio = patch_ratio

    def _binary_mask(self):
        # gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        patch = np.zeros(self.shape)
        patch[self.shape[0] // self.patch_ratio: -self.shape[0] // self.patch_ratio, self.shape[1] // self.patch_ratio: -self.shape[1] // self.patch_ratio] = 1
        return patch.astype(np.uint8)

    def _noise_generate(self):
        temp = np.zeros(self.shape)
        noise = PerlinNoise(octaves=self.octaves)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # temp[i][j] = pnoise2(i / self.shape[0], j / self.shape[1],
                #                      octaves=self.octaves,
                #                      persistence=self.persistence,
                #                      lacunarity=self.lacunarity,
                #                      repeatx=self.shape[0],
                #                      repeaty=self.shape[1])
                temp[i][j] = noise([i / self.shape[0], j / self.shape[1]])
        temp *= 100
        temp = temp.astype(dtype=np.uint8)
        cv2.imwrite(f"{np.random.randint(0, 100)}.png", temp)
        print(temp)
        _, temp = cv2.threshold(temp, 250, 255, cv2.THRESH_BINARY)
        return temp

    def _mask_generate(self):
        m_i = self._binary_mask()
        m_p = self._noise_generate()
        # print(m_i.shape, m_p.shape)
        # print(m_i.dtype, m_p.dtype)
        # print(m_p)

        # print(m_i)
        m_m = cv2.bitwise_and(m_i, m_p)
        return m_m

    def process(self):
        return self._mask_generate()

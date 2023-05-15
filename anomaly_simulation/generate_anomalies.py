from step1 import MaskImage
from step2 import noise_foreground_generate
from step3 import simulated_generate
import os
import glob
import cv2
import os


def generate_anomalies(folder, out_folder, texture_folder, n=100):
    os.makedirs(out_folder, exist_ok=True)
    for image_path in glob.glob(f"{folder}/*")[:n]:
        image_name = image_path.split(os.sep)[-1]
        ori = cv2.imread(image_path)
        mask = MaskImage(image_path).process()
        noise = noise_foreground_generate(ori, mask, texture_folder)
        anomaly = simulated_generate(mask, ori, noise)
        cv2.imwrite(f"{out_folder}/{image_name}", anomaly)


if __name__ == "__main__":
    folder = "/mnt/store/psemchyshyn/cropped_stitches128/val_data"
    out_folder = "/mnt/store/psemchyshyn/cropped_stitches128/simulated_anomalies"
    texture_folder = "/mnt/store/psemchyshyn/dtd/images"
    generate_anomalies(folder, out_folder, texture_folder)
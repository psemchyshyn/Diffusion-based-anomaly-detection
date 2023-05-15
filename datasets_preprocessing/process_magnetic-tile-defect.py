import os
import glob
import cv2
import shutil
import numpy as np

def split_normal_defect(source_path, out_path):
    os.makedirs(out_path, exist_ok=True)

    normal_image_path = os.path.join(out_path, "normal", "images")
    abnormal_image_path = os.path.join(out_path, "abnormal", "images")
    abnormal_mask_path = os.path.join(out_path, "abnormal", "masks")

    os.makedirs(normal_image_path, exist_ok=True)
    os.makedirs(abnormal_image_path, exist_ok=True)
    os.makedirs(abnormal_mask_path, exist_ok=True)

    for dir in os.listdir(f"{source_path}"):
        dir_path = os.path.join(source_path, dir, "Imgs")
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if np.sum(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)) == 0:
                    os.remove(file_path)
                    continue

                if dir.endswith("MT_Free"): # if normal
                    shutil.copy2(file_path, normal_image_path)
                else:
                    if file.endswith(".jpg"):
                        shutil.copy(file_path, os.path.join(abnormal_image_path, f"{dir}_{file}"))
                    else:
                        shutil.copy(file_path, os.path.join(abnormal_mask_path, f"{dir}_{file}"))


if __name__ == "__main__":
    source_path = r"/mnt/store/psemchyshyn/Magnetic-tile-defect-datasets/Magnetic-tile-defect-datasets.-master"
    out_path = "/mnt/store/psemchyshyn/Magnetic-tile-defect-datasets/data"
    split_normal_defect(source_path, out_path)
import os
import glob
import shutil

def split_normal_defect(source_path, out_path):
    os.makedirs(out_path, exist_ok=True)

    normal_image_path = os.path.join(out_path, "normal")
    abnormal_image_path = os.path.join(out_path, "abnormal")

    os.makedirs(normal_image_path, exist_ok=True)
    os.makedirs(abnormal_image_path, exist_ok=True)

    for dir in os.listdir(source_path):
        dir_path = os.path.join(source_path, dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(os.path.join(dir_path, f"U{dir}")):
                file_path = os.path.join(dir_path, f"U{dir}", file)
                shutil.copy2(file_path, normal_image_path)

            for file in os.listdir(os.path.join(dir_path, f"C{dir}")):
                file_path = os.path.join(dir_path, f"C{dir}", file)
                shutil.copy2(file_path, abnormal_image_path)

if __name__ == "__main__":
    source_path = r"/mnt/store/psemchyshyn/SDNET/SDNET_crack_classification"
    out_path = "/mnt/store/psemchyshyn/SDNET/data"
    split_normal_defect(source_path, out_path)
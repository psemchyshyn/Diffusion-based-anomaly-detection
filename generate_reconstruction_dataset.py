import os
import shutil
import glob
from denoising_diffusion_pytorch.diffusion_unconditional.reconstructor_compare_with_emeddings import Reconstructor
from anomaly_detection.utils import get_reconstruct_config
from utils import save_grids


BASE = "/mnt/data/psemchyshyn/diffusion/reconstruction250-1-threshold-0"

config = get_reconstruct_config()
path_model = config["reconstruction"]["model"]["path"]
input_path = config["reconstruction"]["data"]["input_folder"]
out_path = config["reconstruction"]["data"]["output_folder"]
batch_size = config["reconstruction"]["data"]["batch_size"]
normal_folder = config["reconstruction"]["data"]["normal_folder"]
threshold_file_path = config["reconstruction"]["data"]["threshold_file_path"]
threshold_limit = config["reconstruction"]["data"]["threshold_limit"]


classes = ["zipper", "hazelnut", "grid", "screw", "metal_nut", "bottle", "cable", "capsule",
           "carpet", "leather", "pill", "tile", "toothbrush",
           "transistor", "wood"]


basic_threshold_path = "/mnt/data/psemchyshyn/diffusion-info/diffusion-l1-self_condition/reconstruction/placeholder/threshold/300-350/tensor.pt"
basic_normal_folder = "/mnt/data/psemchyshyn/mvtec-diffusion/not_test_data_placeholder"
base_train_folder = "/mnt/data/psemchyshyn/mvtec-diffusion-subnetwork/train"
base_val_folder = "/mnt/data/psemchyshyn/mvtec-diffusion-subnetwork/val"
base_test_folder = "/mnt/data/psemchyshyn/mvtec-diffusion-subnetwork/test"


# Don't forget to fill them with data
all_train_folder = "/mnt/data/psemchyshyn/mvtec-diffusion-subnetwork/train/all"
all_val_folder = "/mnt/data/psemchyshyn/mvtec-diffusion-subnetwork/val/all"
all_test_folder = "/mnt/data/psemchyshyn/mvtec-diffusion/test_data"


rec_train_folder = f"{BASE}/train"
rec_val_folder = f"{BASE}/val"
rec_test_folder = f"{BASE}/test"


def form_threshold_path(path, category):
    splitted = path.split(os.sep)
    path = splitted[:7] + [category] + splitted[8:]
    return f"{os.sep}".join(path)

def form_class_specific_folder(base, clas):
    class_folder = os.path.join(base, clas)
    if not os.path.isdir(class_folder):
        os.makedirs(class_folder, exist_ok=True)

    return class_folder

def copy_class_elements(fromm, to, clas):
    matching_elements = glob.glob(fromm + "/" + f"{clas}*.png")

    for element in matching_elements:
        element_basename = os.path.basename(element)
        target_path = os.path.join(to, element_basename)
        if os.path.isfile(element):
            shutil.copy(element, target_path)

def generate_reconstructions(base_folder, input_folder, out_folder):
    print(f"Reconstructing images from {input_folder} to {out_folder}")
    for category in classes:
        print(f"Reconstructing {category} class...")
        threshold_file_path = form_threshold_path(basic_threshold_path, category)
        threshold_folder = f"{os.sep}".join(threshold_file_path.split(os.sep)[:-1])
        print(f"Threshold file path: {threshold_file_path}")
        class_data_folder = form_class_specific_folder(base_folder, category)
        copy_class_elements(input_folder, class_data_folder, category)

        reconstructor = Reconstructor(path_model, class_data_folder, threshold_file_path, threshold_limit, batch_size)

        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(threshold_folder, exist_ok=True)

        recs = reconstructor.reconstruct(class_data_folder, **config["reconstruction"]["params"])

        names = recs[-1]
        recs = [recs[2]]
        save_grids(out_folder, *recs, names=names, grid_shape=(1, 1))
        print(f"Successfully reconstructed elements to {out_folder}")


if __name__ == "__main__":
    generate_reconstructions(base_val_folder, all_val_folder, rec_val_folder)
    generate_reconstructions(base_train_folder, all_train_folder, rec_train_folder)
    generate_reconstructions(base_test_folder, all_test_folder, rec_test_folder)


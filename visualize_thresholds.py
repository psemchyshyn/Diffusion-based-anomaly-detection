import torch
import torchvision.transforms.functional as TF
import os
import glob
import shutil

classes = ["zipper", "hazelnut", "grid", "screw", "metal_nut", "bottle", "cable", "capsule",
           "carpet", "leather", "pill", "tile", "toothbrush",
           "transistor", "wood"]
#
#
# base = "/mnt/data/psemchyshyn/diffusion-info/diffusion-l1-self_condition"
# temp = "temp"
#
# os.makedirs(temp, exist_ok=True)
#
# for clas in classes:
#     p = os.path.join(base, clas, "placeholder", "threshold", "300-350", "tensor.pt")
#     tensor = torch.load(p)
#     tensor = torch.mean(tensor, dim=0).unsqueeze(0)
#     img = TF.to_pil_image(tensor)
#     img.save(os.path.join(temp, f"{clas}.png"))


base = "/mnt/data/psemchyshyn/mvtec-diffusion"

def copy_class_elements(fromm, to, clas):
    matching_elements = glob.glob(fromm + "/" + f"{clas}*.png")

    for element in matching_elements:
        element_basename = os.path.basename(element)
        target_path = os.path.join(to, element_basename)
        if os.path.isfile(element):
            shutil.copy(element, target_path)


for clas in classes:
    to = os.path.join(base, "splitted_test", clas)
    os.makedirs(to, exist_ok=True)
    copy_class_elements(f"{base}/test_data", to, clas)
    # copy_class_elements(f"{base}/val_data", to, clas)


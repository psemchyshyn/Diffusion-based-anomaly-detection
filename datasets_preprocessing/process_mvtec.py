import os
import random
import shutil

random.seed(42)

def split_mvtec(path_to_mvtec, train_out, val_out, test_out):
    for element_type in os.listdir(path_to_mvtec):
        # print(element_type)
        # element_dir_train = os.path.join(path_to_mvtec, element_type, "train", "good")
        #
        # all = os.listdir(element_dir_train)
        # idx = int(0.8*len(all))
        # train_names = all[:idx]
        # val_names = all[idx:]

        # for name in train_names:
        #     dst = os.path.join(train_out, f"{element_type}-{name}")
        #     src = os.path.join(element_dir_train, name)
        #     shutil.copyfile(src, dst)
        #
        # for name in val_names:
        #     dst = os.path.join(val_out, f"{element_type}-{name}")
        #     src = os.path.join(element_dir_train, name)
        #     shutil.copyfile(src, dst)


        element_dir_test = os.path.join(path_to_mvtec, element_type, "ground_truth")
        for type_test in os.listdir(element_dir_test):
            test_type_folder = os.path.join(element_dir_test, type_test)
            test_names = os.listdir(test_type_folder)

            for name in test_names:
                dst = os.path.join(test_out, f"{element_type}-{type_test}-{name[:-9]}.png")
                src = os.path.join(os.path.join(element_dir_test, type_test), name)
                shutil.copyfile(src, dst)


if __name__ == "__main__":
    path = "/mnt/data/psemchyshyn/mvtec"
    out_train = "/mnt/store/psemchyshyn/mvtec-diffusion/train_data"
    out_val = "/mnt/store/psemchyshyn/mvtec-diffusion/val_data"
    out_test = "/mnt/data/psemchyshyn/mvtec-diffusion/test_data_masks"
    split_mvtec(path, out_train, out_val, out_test)






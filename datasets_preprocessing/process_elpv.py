import os
import pandas as pd
import shutil

def split_normal_defect(data_path, labels_path, out_path, anomaly_threshold=0.05):
    os.makedirs(out_path, exist_ok=True)

    normal_path = os.path.join(out_path, "normal")
    abnormal_path = os.path.join(out_path, "abnormal")

    os.makedirs(normal_path, exist_ok=True)
    os.makedirs(abnormal_path, exist_ok=True)

    labels_df = pd.read_csv(labels_path, sep=r"\s+|\t+|\s+\t+|\t+\s+", header=None)

    for i, val in labels_df.iterrows():
        image_path, anomaly_prob, _ = val.values

        image_path_full = os.path.join(data_path, image_path)
        if anomaly_prob > anomaly_threshold:
            shutil.copy2(image_path_full, abnormal_path)
        else:
            shutil.copy2(image_path_full, normal_path)

if __name__ == "__main__":
    images_path = r"/mnt/store/psemchyshyn/ELPV_dataset_classification/elpv-dataset"
    labels_path = r"/mnt/store/psemchyshyn/ELPV_dataset_classification/elpv-dataset/labels.csv"
    out_path = "/mnt/store/psemchyshyn/ELPV_dataset_classification/data"
    split_normal_defect(images_path, labels_path, out_path)
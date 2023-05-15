import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image as Img
from sklearn import metrics
import cv2
import tqdm
np.random.seed(7)


def get_preds_trues_image_level(all_data, classs):
    all_data_split = []
    for el in all_data:
        if classs not in el:
            continue
        if el[0] == "-":

            el = el[1:].split("-")
            el[0] = "-" + el[0]

        else:
            el = el.split("-")

        all_data_split.append(el)


    all_data_info = [(float(el[0]), el[-2] != "good") for el in all_data_split]
    true = np.array([el[1] for el in all_data_info])
    preds = np.array([el[0] for el in all_data_info])
    return preds, true


def get_auc(preds, true):
    fpr, tpr, thresholds = metrics.roc_curve(true, preds)
    # print(fpr, tpr, thresholds)

    # precision, recall, thresholds_pr = metrics.precision_recall_curve(true, preds)
    aucroc = metrics.auc(fpr, tpr)
    return aucroc


FOLDER = r"/mnt/data/psemchyshyn/diffusion-info/diffusion-l1-self_condition/reconstruction/test_data/rec/without-reconstructed-50-1-threshold-0"
all_data = os.listdir(FOLDER)
scores = {}
classes = ["tile", "grid", "leather", "wood", "carpet", "bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]
for classs in classes:
    aucroc = get_auc(*get_preds_trues_image_level(all_data, classs))
    scores[classs] = aucroc


print(scores)
print(np.mean([score for score in scores.values()]))
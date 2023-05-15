import os

from flask import Flask, request
import numpy as np
from anomaly_detection.metrics import MSE
from utils import get_reconstructed_score, save_grids
from distribution_utils import get_predictor_from_data
from anomaly_detection.init_model import create_model
import yaml

app = Flask(__name__)
metrics_dct = {"mse": MSE()}


with open("anomaly_detection/config.yaml", 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

config["model"]["data"]["results_folder"] = "/mnt/store/psemchyshyn/checkpoints/rivets/checkpoints_64_augs_ds_yak_20190416_nau_fuselage-20190909"
config["model"]["params"]["image_size"] = 64

rivet_model = create_model(config)
config["model"]["data"]["results_folder"] = "/mnt/store/psemchyshyn/checkpoints/stitches/checkpoints_128_ds_yak_20190416_nau_fuselage-20190909"
config["model"]["params"]["image_size"] = 128
stitches_model = create_model(config)

@app.route("/reconstruct-score", methods=["POST"])
def reconstruct_score():
    folder = request.json["folder"]
    out_folder = request.json["results_folder"]
    score_path = request.json["scores_path"]
    strategy = request.json["strategy"]
    model_epoch = request.json["model_epoch"]
    diff_steps = request.json["diff_steps"]
    resampling_steps = request.json["resampling_steps"]
    patch_ratio = request.json["patch_ratio"]
    fill_val = request.json["fill_val"]
    patch_only = request.json["patch_only"]
    metric_str = request.json["metric"]
    is_rivet = request.json["is_rivet"]

    if not metric_str in metrics_dct:
        metric = MSE()
    else:
        metric = metrics_dct[metric_str]

    if is_rivet:
        model = rivet_model
    else:
        model = stitches_model
    model.load(model_epoch)

    orig, cropped, rec, diff = get_reconstructed_score(folder, model, strategy, patch_ratio, fill_val, diff_steps, resampling_steps, metric=metric, patch_only=patch_only, save_to_scores=score_path)
    files = os.listdir(folder)
    # print(files)
    save_grids(out_folder, orig, cropped, rec, diff, names=files)
    return "OK"

@app.route("/rate-anomaly", methods=["POST"])
def rate_anomaly():
    scores_data_path = request.json["scores_path"]
    anomaly_scores_out = request.json["anomaly_scores_path"]
    is_rivet = request.json["is_rivet"]

    if is_rivet:
            dist_path = "/home/psemchyshyn/projects/diffusion-reconstruction/rivets/with_augs/scores_on_patch_size4_resampling"
    else:
        dist_path = "/home/psemchyshyn/projects/diffusion-reconstruction/stitches/without_augs/scores_on_patch_size4_resampling"

    predictor = get_predictor_from_data(dist_path)
    scores = np.load(scores_data_path)
    anomaly_scores = []
    for el in scores:
        anomaly_scores.append(predictor(el))

    print(anomaly_scores)

    np.save(anomaly_scores_out, np.array(anomaly_scores))
    return "OK"


if __name__ == "__main__":
    app.run(port=8082, debug=True)

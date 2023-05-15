import os

from segmentation.predictor import Predictor
from utils import save_grids
import yaml

def get_reconstruct_config(config_path="segmentation/predictor-config.yaml"):
    config = None
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return config

if __name__ == "__main__":
    config = get_reconstruct_config()

    path_to_model = config["path_to_model"]
    out_path = config["out_path"]
    input_path = config["input_path"]
    reconstructor = Predictor(path_to_model, batch_size=8)

    os.makedirs(out_path, exist_ok=True)

    # recs = reconstructor.reconstruct(input_path)
    recs = reconstructor.calculate_anomaly_score(input_path)

    names = recs[-1]
    recs = recs[:-1]
    save_grids(out_path, *recs, names=names, grid_shape=(1, 1))

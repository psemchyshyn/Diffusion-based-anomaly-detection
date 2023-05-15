import os

from classification.predictor import Predictor
from utils import save_grids


if __name__ == "__main__":
    path_to_model = "/mnt/data/psemchyshyn/checkpoints/classification/cross-entropy-10-5-threshold-5-add/checkpoints_256/last.ckpt"
    out_path = "/mnt/data/psemchyshyn/diffusion-info/diffusion-classification/reconstruction/test_data/cross-entropy10-5-threshold-5-add"

    reconstructor = Predictor(path_to_model, batch_size=64)

    os.makedirs(out_path, exist_ok=True)

    # recs = reconstructor.reconstruct(input_path)
    recs = reconstructor.calculate_anomaly_score()

    names = recs[-1]
    recs = recs[:-1]
    save_grids(out_path, *recs, names=names, grid_shape=(2, 2))

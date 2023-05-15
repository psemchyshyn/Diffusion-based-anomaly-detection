import os

# from denoising_diffusion_pytorch.diffusion_unconditional.reconstructor import Reconstructor
# from denoising_diffusion_pytorch.diffusion_simulated_anomalies.reconstructor import Reconstructor
from denoising_diffusion_pytorch.diffusion_unconditional.reconstructor_compare_with_emeddings import Reconstructor
from anomaly_detection.utils import get_reconstruct_config
from utils import save_grids

if __name__ == "__main__":
    config = get_reconstruct_config()

    path_model = config["reconstruction"]["model"]["path"]
    input_path = config["reconstruction"]["data"]["input_folder"]
    out_path = config["reconstruction"]["data"]["output_folder"]
    batch_size = config["reconstruction"]["data"]["batch_size"]
    normal_folder = config["reconstruction"]["data"]["normal_folder"]
    threshold_file_path = config["reconstruction"]["data"]["threshold_file_path"]
    noise_threshold_from = config["reconstruction"]["data"]["threshold_file_path"]
    threshold_limit = config["reconstruction"]["data"]["threshold_limit"]

    threshold_folder = f"{os.sep}".join(threshold_file_path.split(os.sep)[:-1])
    reconstructor = Reconstructor(path_model, normal_folder, threshold_file_path, threshold_limit, batch_size)

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(threshold_folder, exist_ok=True)

    recs = reconstructor.calculate_image_level_anomaly_score(input_path, **config["reconstruction"]["params"])

    names = recs[-1]
    recs = recs[:-1]
    save_grids(out_path, *recs, names=names, grid_shape=(1, 5))

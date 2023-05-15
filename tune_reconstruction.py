import torchmetrics
from anomaly_detection.init_model import create_model, config
from anomaly_detection.metrics import *
import math


class Tuner:
    def __init__(self, trainer, folder, patch_ratio, metrics_lst=None, weights=None):
        self.trainer = trainer
        self.folder = folder
        self.metrics_lst = metrics_lst
        self.patch_ratio = patch_ratio
        if weights is None:
            self.weights_lst = [1]*len(self.metrics_lst)
        else:
            self.weights_lst = weights

    def set_metrics(self, metrics):
        self.metrics_lst = metrics
        if len(self.weights_lst) != len(self.metrics_lst):
            self.weights_lst = [1]*len(self.metrics_lst)

    def set_weights(self, metric_weights: list):
        self.weights_lst = metric_weights

    def tune_reconstruction(self, parameter_space, strategy, patch_only=False):
        assert len(self.metrics_lst) == len(
            self.weights_lst), f"Metrics and their weights are of different size, there are {len(self.metrics_lst)} metrics and {len(self.weights_lst)} weights"
        steps = parameter_space["steps"]
        noises = parameter_space["noise_amount"]
        fill_patch_vals = parameter_space["fill_patch_val"]
        resampling_num = parameter_space["resampling"]
        best_score = math.inf
        best_params = None
        # grid search for finding optimal parameters
        for i in range(len(steps)):
            for j in range(len(noises)):
                for q in range(len(fill_patch_vals)):
                    for w in range(len(resampling_num)):
                        step = steps[i]
                        noise = noises[j]
                        fill_patch_val = fill_patch_vals[q]
                        resampling = resampling_num[w]

                        res = self.get_one_step_metric(step, noise, fill_patch_val, resampling, strategy, patch_only)
                        acc = sum(map(lambda metric, weight: metric * weight, res.values(), self.weights_lst)).item()
                        print(
                            f"Step: {step}, noise: {noise}, fill_patch_val: {fill_patch_val}, resampling: {resampling} - {res}. Score={acc}")

                        if acc > best_score:
                            best_score = acc
                            best_params = (step, noise, fill_patch_val, resampling)
        return best_score, best_params

    def get_one_step_metric(self, step, noise, fill_patch_val, resampling=10, strategy="resampling", patch_only=False):
        self.trainer.load(step)

        dct = {
            "fill_value": fill_patch_val,
            "noise": noise,
            "resampling": resampling,
            "patch_ratio":  self.patch_ratio
        }

        images_or, *_, images_rec = self.trainer.reconstruct(self.folder, strategy, **dct)

        if patch_only:
            _, _, height, width = images_or.shape
            images_or = images_or[:, :, height // self.patch_ratio: -height // self.patch_ratio, width // self.patch_ratio: -width // self.patch_ratio]
            images_rec = images_rec[:, :, height // self.patch_ratio: -height // self.patch_ratio, width // self.patch_ratio: -width // self.patch_ratio]
        metr_res = self.metrics_lst(images_rec.cpu(), images_or.cpu())

        return metr_res


if __name__ == "__main__":
    folder = config["reconstruction"]["data"]["folder"]
    patch_ratio = 6 # 1/8 part of the image is cropped

    space = {
        "steps": [400],
        "noise_amount": [i for i in range(100, 500, 50)],
        "fill_patch_val": ["mean"],
        "resampling": [10]
    }

    metrics = torchmetrics.MetricCollection([SSIM()])

    model = create_model(config)
    tuner = Tuner(model, folder, patch_ratio, metrics)
    score, params = tuner.tune_reconstruction(space, "resampling", patch_only=False)
    print(score, params)

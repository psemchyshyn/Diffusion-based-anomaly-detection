import os

import torch
import tqdm
from torch.utils import data
from segmentation.discriminative_lightning import SegLit
from segmentation.dataset import DiffusionDataset

class Predictor:
    def __init__(self, path_to_lit_model, batch_size=None):
        self.lit = SegLit.load_from_checkpoint(path_to_lit_model).cuda()
        self.image_size = self.lit.image_size
        self.channels = self.lit.channels
        self.batch_size = self.lit.batch_size if not batch_size else batch_size
        self.lit.eval()


    def calculate_anomaly_score(self, folder_to_score):
        orig, reconstruction, predictions, names = self.get_predictions(folder_to_score)
        image_level, _ = predictions.view(predictions.size(0), -1).max(dim=-1)
        # predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
        # predictions = torch.stack(predictions, dim=0)
        names = [f"{score.item()}-{name}" for score, name in zip(image_level, names)]
        return [predictions, names]

    def reconstruct(self, folder_to_score):
        orig, reconstruction, predictions, names = self.get_predictions(folder_to_score)
        image_level, _ = predictions.view(predictions.size(0), -1).max(dim=-1)
        # predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
        predictions = (predictions > 0.5).float()

        names = [f"{score.item()}-{name}" for score, name in zip(image_level, names)]
        return [orig, reconstruction, predictions, names]


    def get_predictions(self, folder):
        torch.set_grad_enabled(False)
        dl = data.DataLoader(DiffusionDataset(folder, self.image_size, self.channels), batch_size=self.batch_size,
                             shuffle=False,
                             pin_memory=True)

        preds_lst = []
        orig_lst = []
        rec_lst = []
        names = []
        for idx, batch in tqdm.tqdm(enumerate(dl)):
            batch["image"] = batch["image"].cuda()
            reconstruction, predictions = self.lit(batch)
            preds_lst.append(predictions)
            orig_lst.append(batch["image"])
            rec_lst.append(reconstruction)
            names.extend([path.split(os.sep)[-1] for path in batch["path"]])
        return [torch.cat(orig_lst, dim=0), torch.cat(rec_lst, dim=0), torch.cat(preds_lst, dim=0), names]

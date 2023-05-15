import os

import torch
import tqdm
from classification.discriminative_lightning import ClassifierLit

class Predictor:
    def __init__(self, path_to_lit_model, batch_size=None):
        self.lit = ClassifierLit.load_from_checkpoint(path_to_lit_model).cuda()
        self.image_size = self.lit.image_size
        self.channels = self.lit.channels
        self.batch_size = self.lit.batch_size if not batch_size else batch_size
        self.lit.eval()


    def calculate_anomaly_score(self):
        orig, reconstruction, predictions, names = self.get_predictions_test()
        # predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
        # predictions = torch.stack(predictions, dim=0)
        names = [f"{round(score.item(), 3)}-{name}" for score, name in zip(predictions, names)]
        return [orig, reconstruction, names]

    def get_predictions_test(self):
        torch.set_grad_enabled(False)
        dl = self.lit.test_dataloader()

        preds_lst = []
        orig_lst = []
        rec_lst = []
        names = []
        for idx, batch in tqdm.tqdm(enumerate(dl)):
            batch["image"] = batch["image"].cuda()
            batch["reconstruction"] = batch["reconstruction"].cuda()
            predictions = self.lit(batch)
            preds_lst.append(predictions)
            orig_lst.append(batch["image"])
            rec_lst.append(batch["reconstruction"])
            names.extend([path.split(os.sep)[-1] for path in batch["path"]])
        return [torch.cat(orig_lst, dim=0), torch.cat(rec_lst, dim=0), torch.cat(preds_lst, dim=0), names]

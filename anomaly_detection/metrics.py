import torch
import torchmetrics
import einops

class SSIM(torchmetrics.StructuralSimilarityIndexMeasure):
    def __init__(self, *args, **kwargs):
        super(SSIM, self).__init__(*args, **kwargs)

    def update(self, preds, target):
        preds = preds*255
        target = target*255
        super().update(preds, target)

#
# class FID(torchmetrics.image.fid.FrechetInceptionDistance):
#     def __init__(self, *args, **kwargs):
#         super(FID, self).__init__(*args, **kwargs)
#
#     def forward(self, rec, orig):
#         # if image is one-channel
#         # rec = einops.repeat(rec, 'b c h w -> b (repeat c) h w', repeat=3)
#         # orig = einops.repeat(orig, 'b c h w -> b (repeat c) h w', repeat=3)
#         self.update((rec*255).to(torch.uint8), real=False)
#         self.update((orig*255).to(torch.uint8), real=True)
#         result = self.compute()
#         self.reset()
#         return result

class MSE(torchmetrics.MeanSquaredError):
    def __init__(self, *args, **kwargs):
        super(MSE, self).__init__(*args, **kwargs)

    def forward(self, rec, orig):
        self.reset()
        return super().forward(rec*255, orig*255)

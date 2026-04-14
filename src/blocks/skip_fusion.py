import torch
import torch.nn as nn


class CenterCropLike(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src, ref):
        _, _, h, w = ref.shape
        _, _, hs, ws = src.shape

        top = (hs - h) // 2
        left = (ws - w) // 2

        return src[:, :, top:top + h, left:left + w]


class SkipFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop = CenterCropLike()

    def forward(self, upsampled, skip):
        skip = self.crop(skip, upsampled)
        return upsampled + skip

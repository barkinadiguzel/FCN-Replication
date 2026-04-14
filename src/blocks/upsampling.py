import numpy as np
import torch
import torch.nn as nn


class BilinearUpsample(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self._init_bilinear()

    def _make_bilinear_weights(self, channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - np.abs(og[0] - center) / factor) * (1 - np.abs(og[1] - center) / factor)

        weight = np.zeros((channels, channels, kernel_size, kernel_size), dtype=np.float32)
        for c in range(channels):
            weight[c, c, :, :] = filt

        return torch.from_numpy(weight)

    def _init_bilinear(self):
        with torch.no_grad():
            w = self._make_bilinear_weights(self.deconv.in_channels, self.deconv.kernel_size[0])
            self.deconv.weight.copy_(w)

    def forward(self, x):
        return self.deconv(x)

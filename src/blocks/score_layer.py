import torch.nn as nn


class ScoreLayer(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.layer = nn.Conv2d(in_ch, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.layer(x)

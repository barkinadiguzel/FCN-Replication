import torch.nn as nn


class FC6Conv(nn.Module):
    def __init__(self, in_ch=512, out_ch=4096):
        super().__init__()
        self.layer = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=1, padding=0)

    def forward(self, x):
        return self.layer(x)


class FC7Conv(nn.Module):
    def __init__(self, in_ch=4096, out_ch=4096):
        super().__init__()
        self.layer = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.layer(x)

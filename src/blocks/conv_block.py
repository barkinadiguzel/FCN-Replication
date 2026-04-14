import torch.nn as nn

def conv_block(in_ch, out_ch, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding),
        nn.ReLU(inplace=True)
    )

import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGG16FCNBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = vgg16(weights=weights)
        feats = vgg.features

        self.block1 = feats[:5]
        self.block2 = feats[5:10]
        self.block3 = feats[10:17]
        self.block4 = feats[17:24]
        self.block5 = feats[24:31]

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        pool3 = self.block3(x)
        pool4 = self.block4(pool3)
        pool5 = self.block5(pool4)
        return pool3, pool4, pool5

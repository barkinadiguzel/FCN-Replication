import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

from src.backbone.vgg16_fcn import VGG16FCNBackbone
from src.blocks.fc_to_conv import FC6Conv, FC7Conv
from src.blocks.score_layer import ScoreLayer
from src.blocks.skip_fusion import CenterCropLike, SkipFusion
from src.blocks.upsampling import BilinearUpsample


class FCN8s(nn.Module):
    def __init__(self, num_classes=21, pretrained_backbone=True, load_vgg16_classifier_weights=True):
        super().__init__()

        # Backbone
        self.backbone = VGG16FCNBackbone(pretrained=pretrained_backbone)

        # FC -> Conv
        self.fc6 = FC6Conv(512, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout(p=0.5)

        self.fc7 = FC7Conv(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout(p=0.5)

        # Score layers
        self.score_fr = ScoreLayer(4096, num_classes)
        self.score_pool4 = ScoreLayer(512, num_classes)
        self.score_pool3 = ScoreLayer(256, num_classes)

        # Upsampling (deconv)
        self.upscore2 = BilinearUpsample(num_classes, kernel_size=4, stride=2, padding=1)
        self.upscore_pool4 = BilinearUpsample(num_classes, kernel_size=4, stride=2, padding=1)
        self.upscore8 = BilinearUpsample(num_classes, kernel_size=16, stride=8, padding=4)

        # Utils
        self.crop = CenterCropLike()
        self.fuse = SkipFusion()

        self._init_score_layers_zero()

        if load_vgg16_classifier_weights:
            self._load_vgg16_classifier_weights()

    def _init_score_layers_zero(self):
        for layer in [self.score_fr.layer, self.score_pool4.layer, self.score_pool3.layer]:
            nn.init.constant_(layer.weight, 0.0)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def _load_vgg16_classifier_weights(self):
        weights = VGG16_Weights.IMAGENET1K_V1
        vgg = vgg16(weights=weights)

        with torch.no_grad():
            self.fc6.layer.weight.copy_(vgg.classifier[0].weight.data.view(4096, 512, 7, 7))
            self.fc6.layer.bias.copy_(vgg.classifier[0].bias.data)

            self.fc7.layer.weight.copy_(vgg.classifier[3].weight.data.view(4096, 4096, 1, 1))
            self.fc7.layer.bias.copy_(vgg.classifier[3].bias.data)

    def forward(self, x):
        input_ref = x

        # backbone
        pool3, pool4, pool5 = self.backbone(x)

        # classifier head
        x = self.fc6(pool5)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        score_fr = self.score_fr(x)

        # FCN-32s → FCN-16s
        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(pool4)
        score_pool4 = self.crop(score_pool4, upscore2)
        fuse_pool4 = self.fuse(upscore2, score_pool4)

        # FCN-16s → FCN-8s
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score_pool3 = self.score_pool3(pool3)
        score_pool3 = self.crop(score_pool3, upscore_pool4)
        fuse_pool3 = self.fuse(upscore_pool4, score_pool3)

        # final upsample
        out = self.upscore8(fuse_pool3)
        out = self.crop(out, input_ref)

        return out

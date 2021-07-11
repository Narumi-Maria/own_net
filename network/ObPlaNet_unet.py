"""仅对背景使用Unet模型"""
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image


import sys
sys.path.append("..")
from network.BaseBlocks import BasicConv2d
from network.tensor_ops import cus_sample, upsample_add

from network.MyModules import (
    DDPM,
    DenseTransLayer,
)

from network.OwnModules import simpleDFN

from backbone.ResNet import Backbone_ResNet50_in1, Backbone_ResNet50_in3, Backbone_ResNet18_in1, Backbone_ResNet18_in3, Backbone_ResNet18_in3_1


class ObPlaNet_resnet18(nn.Module):
    def __init__(self, pretrained=True):
        super(ObPlaNet_resnet18, self).__init__()
        self.Eiters = 0
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        self.to_pil = transforms.ToPILImage()

        (
            self.bg_encoder1,
            self.bg_encoder2,
            self.bg_encoder4,
            self.bg_encoder8,
            self.bg_encoder16,
        ) = Backbone_ResNet18_in3(pretrained=pretrained)

        self.upconv16 = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Conv2d(64, 2, 1)

    def forward(self, bg_in_data, mode='val'):
        """
        Args:
            bg_in_data: (batch_size * 3 * H * W) 背景图片特征
            mode: 当前模式 train / val, 前者为训练, 后者为测试
        """
        if('train' == mode):
            self.Eiters += 1
        bg_in_data_1 = self.bg_encoder1(bg_in_data)
        del bg_in_data

        bg_in_data_2 = self.bg_encoder2(bg_in_data_1)
        bg_in_data_4 = self.bg_encoder4(bg_in_data_2)

        bg_in_data_8 = self.bg_encoder8(bg_in_data_4)
        bg_in_data_16 = self.bg_encoder16(bg_in_data_8)

        bg_out_data_16 = bg_in_data_16
        bg_out_data_8 = self.upsample_add(self.upconv16(bg_out_data_16), bg_in_data_8)
        bg_out_data_4 = self.upsample_add(self.upconv8(bg_out_data_8), bg_in_data_4)
        bg_out_data_2 = self.upsample_add(self.upconv4(bg_out_data_4), bg_in_data_2)
        bg_out_data_1 = self.upsample_add(self.upconv2(bg_out_data_2), bg_in_data_1)
        del bg_out_data_2, bg_out_data_4, bg_out_data_8, bg_out_data_16

        bg_out_data = self.upconv1(self.upsample(bg_out_data_1, scale_factor=2))

        out_data = self.classifier(bg_out_data)

        return out_data

if __name__ == "__main__":
    a = torch.randn((2, 3, 256, 256))
    b = torch.randn((2, 3, 256, 256))
    c = torch.randn((2, 1, 256, 256))

    model = ObPlaNet_resnet18()
    x = model(a, c)
    print(x.size())

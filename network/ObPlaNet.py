import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image

from network.BaseBlocks import BasicConv2d
from network.tensor_ops import cus_sample, upsample_add

from network.MyModules import (
    DDPM,
    DenseTransLayer,
)

from backbone.ResNet import Backbone_ResNet50_in1, Backbone_ResNet50_in3
from backbone.VGG import (
    Backbone_VGG19_in1,
    Backbone_VGG19_in3,
    Backbone_VGG_in1,
    Backbone_VGG_in3,
)


class ObPlaNet_VGG19(nn.Module):
    def __init__(self, pretrained=True):
        super(ObPlaNet_VGG19, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        self.to_pil = transforms.ToPILImage()

        (
            self.bg_encoder1,
            self.bg_encoder2,
            self.bg_encoder4,
            self.bg_encoder8,
            self.bg_encoder16,
        ) = Backbone_VGG19_in3(pretrained=pretrained)
        (
            self.fg_encoder1,
            self.fg_encoder2,
            self.fg_encoder4,
            self.fg_encoder8,
            self.fg_encoder16,
        ) = Backbone_VGG19_in3(pretrained=pretrained)


        self.trans16 = nn.Conv2d(512, 64, 1)
        self.trans8 = nn.Conv2d(512, 64, 1)
        self.trans4 = nn.Conv2d(256, 64, 1)
        self.trans2 = nn.Conv2d(128, 64, 1)
        self.trans1 = nn.Conv2d(64, 32, 1)

        self.fg_trans16 = DenseTransLayer(512, 64)
        self.fg_trans8 = DenseTransLayer(512, 64)
        self.fg_trans4 = DenseTransLayer(256, 64)

        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_4 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 2, 1)
        # self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, bg_in_data, fg_in_data):
        bg_in_data_1 = self.bg_encoder1(bg_in_data)
        del bg_in_data
        fg_in_data_1 = self.fg_encoder1(fg_in_data)
        del fg_in_data

        bg_in_data_2 = self.bg_encoder2(bg_in_data_1)
        fg_in_data_2 = self.fg_encoder2(fg_in_data_1)
        bg_in_data_4 = self.bg_encoder4(bg_in_data_2)
        fg_in_data_4 = self.fg_encoder4(fg_in_data_2)
        del fg_in_data_1, fg_in_data_2

        bg_in_data_8 = self.bg_encoder8(bg_in_data_4)
        fg_in_data_8 = self.fg_encoder8(fg_in_data_4)
        bg_in_data_16 = self.bg_encoder16(bg_in_data_8)
        fg_in_data_16 = self.fg_encoder16(fg_in_data_8)

        in_data_4_aux = self.fg_trans4(bg_in_data_4, fg_in_data_4)
        in_data_8_aux = self.fg_trans8(bg_in_data_8, fg_in_data_8)
        in_data_16_aux = self.fg_trans16(bg_in_data_16, fg_in_data_16)
        del fg_in_data_4, fg_in_data_8, fg_in_data_16

        bg_in_data_1 = self.trans1(bg_in_data_1)
        bg_in_data_2 = self.trans2(bg_in_data_2)
        bg_in_data_4 = self.trans4(bg_in_data_4)
        bg_in_data_8 = self.trans8(bg_in_data_8)
        bg_in_data_16 = self.trans16(bg_in_data_16)

        out_data_16 = bg_in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), bg_in_data_8)
        del out_data_16, in_data_16_aux, bg_in_data_8

        out_data_8 = self.upconv8(out_data_8)  # 512
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), bg_in_data_4)
        del out_data_8, in_data_8_aux, bg_in_data_4

        out_data_4 = self.upconv4(out_data_4)  # 256
        out_data_2 = self.upsample_add(self.selfdc_4(out_data_4, in_data_4_aux), bg_in_data_2)
        del out_data_4, in_data_4_aux, bg_in_data_2

        out_data_2 = self.upconv2(out_data_2)  # 64
        out_data_1 = self.upsample_add(out_data_2, bg_in_data_1)
        del out_data_2, bg_in_data_1

        out_data_1 = self.upconv1(out_data_1)  # 32

        out_data = self.classifier(out_data_1)

        return out_data.sigmoid()

    def get_point(self, pred, pos_xs, pos_ys, ori_ws, ori_hs):
        pred_array_tensor = pred

        pre_pos = torch.zeros(pred.size()[0], requires_grad=True)

        for item_id, pred_tensor in enumerate(pred_array_tensor):
            print(pred_tensor.size)
            pred_array = F.interpolate(pred_tensor, [ori_ws, ori_hs], mode='bilinear', align_corners=True)
            pre_pos[item_id] = pred_array[pos_xs[item_id]][pos_ys[item_id]]

        return pre_pos

if __name__ == "__main__":
    a = torch.randn((2, 3, 320, 320))
    b = torch.randn((2, 3, 320, 320))

    model = ObPlaNet_VGG19()
    x = model(a,b, [5,5], [4,6])
    print(x.size())

import os

import csv
import json
import torch
import numpy as np
from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
sys.path.append("..")
from data.all_transforms import Compose, JointResize

class CPDataset(Dataset):
    def __init__(self, file, coco_dir, fg_dir, mask_dir, in_size, datatype = 'train'):
        """
        初始化数据集

        Args:
            file(str): 测试/训练数据信息存储的文件路径,
            coco_dir(str): 背景图片存放的文件夹,
            fg_dir(str): 背景图片存放的文件夹,
            mask_dir(str): 背景图片存放的文件夹,
            in_size(int): 图片resize的大小,
            datatype(str): train/val, 指定加载的是训练集还是测试集,
        """
        # 从文件中加载数据信息
        self.data = _collect_info(file, coco_dir, fg_dir, mask_dir, datatype)
        self.insize = in_size

        # 对图片的处理
        self.train_triple_transform = Compose([JointResize(in_size)])
        self.train_img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
            ]
        )
        self.train_mask_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        数据集加载单条数据
        return:
            i: 图片在数据集中的序号,
            bg_t:(1 * 3 * in_size * in_size)背景图片特征,
            mask_t:(1 * 1 * in_size * in_size)前景mask图片特征,
            bg_t:(1 * 3 * in_size * in_size)前景物体图片特征,
            target_t: (1 * in_size * in_size)对应GT标注的目标,
            labels_num: (int) 该大小的前景-背景组合中有标注的位置数量
        """
        i, bg_path, fg_path, mask_path, scale, pos_label, neg_label = self.data[index]

        bg_img = Image.open(bg_path)
        fg_img = Image.open(fg_path)
        mask = Image.open(mask_path)
        if len(bg_img.split()) != 3:
            bg_img = bg_img.convert("RGB")
        if len(fg_img.split()) == 3:
            fg_img = fg_img.convert("RGB")
        if len(mask.split()) == 3:
            mask = mask.convert("L")

        # 加载相应目标图, 合理位置为1, 不合理位置为0, 其余位置为255
        target = _obtain_target(bg_img.size[0], bg_img.size[1], self.insize, pos_label, neg_label)

        bg_t, fg_t, mask_t= self.train_triple_transform(bg_img, fg_img, mask)
        mask_t = self.train_mask_transform(mask_t)
        fg_t = self.train_img_transform(fg_t)
        bg_t = self.train_img_transform(bg_t)
        target_t = self.train_mask_transform(target) * 255
        labels_num = (target_t != 255).sum()

        return i, bg_t, mask_t, fg_t, target_t.squeeze(), labels_num

def _obtain_target(original_width, original_height, in_size, pos_label, neg_label):
    """
    获得GT目标
    Args:
        original_width(int): 背景图原宽度
        original_height(int): 背景图原高度
        in_size(int): 背景图resize后的大小
        pos_label(list): 有合理标注的原前景中心位置
        neg_label(list): 有不合理标注的原前景中心位置
    return:
        target_r: 对应GT标注的目标
    """
    target = np.uint8(np.ones((in_size, in_size)) * 255)
    for pos in pos_label:
        x, y = pos
        x_new = int(x * in_size / original_width)
        y_new = int(y * in_size / original_height)
        target[x_new, y_new] = 1.
    for pos in neg_label:
        x, y = pos
        x_new = int(x * in_size / original_width)
        y_new = int(y * in_size / original_height)
        target[x_new, y_new] = 0.
    target_r = Image.fromarray(target)
    return target_r

def _collect_info(json_file, coco_dir, fg_dir, mask_dir, datatype = 'train'):
    """
    加载json文件, 返回数据信息以及相应路径
    Args:
        json_file(str): 测试/训练数据信息存储的文件路径,
        coco_dir(str): 背景图片存放的文件夹,
        fg_dir(str): 背景图片存放的文件夹,
        mask_dir(str): 背景图片存放的文件夹,
        datatype(str): train/val, 指定加载的是训练集还是测试集,
    return:
        index(int): 数据在json文件的序号,
        背景图片的路径, 前景物体图片的路径, 前景mask图片路径,
        前景图片scale, 合理的前景位置中心点坐标, 不合理的前景位置中心点坐标
    """
    f_json = json.load(open(json_file, 'r'))
    return [
        (
            index,
            os.path.join(coco_dir, "%012d.jpg" % int(row['scID'])),    #background image path
            os.path.join(fg_dir, "{}/{}_{}_{}_{}.jpg".format(datatype, int(row['annID']), int(row['scID']), int(row['newWidth']), int(row['newHeight']))),  #composite image path, to obtain foreground object
            os.path.join(mask_dir, "{}/{}_{}_{}_{}.jpg".format(datatype, int(row['annID']), int(row['scID']), int(row['newWidth']), int(row['newHeight']))),  #mask image path
            row['scale'],
            row['pos_label'], row['neg_label']
        )
        for index, row in enumerate(f_json)
    ]

def _to_center(bbox):
    """conver bbox to center pixel"""
    x, y, width, height = bbox
    return x + width // 2, y + height // 2

def create_loader(table_path, coco_dir, fg_dir, mask_dir, in_size, datatype, batch_size, num_workers, shuffle):
    dset = CPDataset(table_path, coco_dir, fg_dir, mask_dir, in_size, datatype)
    data_loader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return data_loader

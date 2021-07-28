# @Time    : 2020/7/8
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : config.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang
import os

__all__ = ["proj_root", "arg_config"]

proj_root = os.path.dirname(__file__)
datasets_root = "data/data"

tr_data_path = os.path.join(datasets_root, "train_pair_new.json")  # 训练集数据路径
ts_data_path = os.path.join(datasets_root, "test_pair_new.json")  # 测试集数据路径
# coco_dir = '/media/user/73b6ed39-9723-42c5-8680-3282cb804c62/code/coco/images/train2017' # 合成图中背景图片路径
coco_dir = 'data/data/train2017'
fg_dir = os.path.join(datasets_root, "fg")  # 合成图中前景图片路径
mask_dir = os.path.join(datasets_root, "mask")  # 合成图中前景mask路径

# 配置区域 #####################################################################
arg_config = {
    # 常用配置
    "model": "ObPlaNet_resnet18",  # 模型名字, 注意修改network.__init__文件夹下的import
    "suffix": "simple_mask_adam",
    "resume": False,  # 是否需要恢复模型
    "save_pre": True,  # 是否保留最终的预测结果
    "epoch_num": 45,  # 训练周期
    "lr": 0.0005,
    "tr_data_path": tr_data_path,
    "ts_data_path": ts_data_path,
    "coco_dir": coco_dir,
    "fg_dir": fg_dir,
    "mask_dir": mask_dir,

    "print_freq": 10,  # >0, 保存迭代过程中的信息
    "prefix": (".jpg", ".png"),
    # img_prefix, gt_prefix，用在使用索引文件的时候的对应的扩展名，这里使用的索引文件不包含后缀
    "reduction": "mean",  # 损失处理的方式，可选“mean”和“sum”
    "optim": "Adam_trick",  # 自定义部分的学习率
    "weight_decay": 5e-4,  # 微调时设置为0.0001
    "momentum": 0.9,
    "nesterov": False,
    "lr_type": "all_decay",  # 学习率调整的策略
    "lr_decay": 0.9,  # poly
    "batch_size": 2,
    "num_workers": 6,
    "input_size": 256,  # 图片resize的大小
    "Experiment_name": "Model2+FM(shuffle f)"
}

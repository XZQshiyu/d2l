# 导入所需的包或模块
import os
import torch
import torchvision
import collections
import math
import pandas as pd
from torch import nn
from d2l import torch as d2l

# 下载数据集
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip', '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# 如果已经下载了数据集，我们将使用本地数据集
test = False
if test:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../data/cifar10'

# 整理数据集

def read_csv_labels(fname):
    """读取 `fname` 来给标签字典返回一个CSV文件。"""
    with open(fname, 'r') as f:
        # 跳过文件头行（列名）
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# 训练样本: ', len(labels))
print('# 类别: ', len(set(labels.values())))
print('# 样本: ', len(os.listdir(os.path.join(data_dir, 'train'))))
# coding: utf8

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from utils import plot_image, plot_curve, one_hot
from matplotlib import pyplot as plt
import cnn
from cnn import ResNet18
import MyDataloader
from MyDataloader import MyDataset


import torch
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np

from matplotlib import pyplot as plt

######测试自己的单张图片######

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('HRG_python.pth',map_location='cpu')  # 加载模型

    print("load model success!")

    model = model.to(device)

    model.eval()  # 把模型转为test模式

    ##
    img = cv2.imread("4_0000849.jpeg")  # 读取要预测的图片
    trans = transforms.Compose(
        [
            transforms.ToTensor(),  #归一化到了[0,1]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) #归一化到流[-1,1]
        ])

    img = trans(img)    #这里的transform应该和训练时图片的预处理要一样才可以，在work.py中transform是专为Tensor即可
    img = img.to(device)
    img1 = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    # 扩展后，为[1，3，32，32]
    output = model(img1)
    _, pred1 = torch.max(output, 1)
    print("the predicted hand_result is ", pred1.item())


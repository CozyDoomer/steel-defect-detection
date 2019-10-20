import numpy as np
import pandas as pd
import os

from timeit import default_timer as timer
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.utils.data as data
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
import torch

PI = np.pi
IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]
DEFECT_COLOR = [(0,0,0),(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

# Model Definition
class ConvBn2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ConvBn2dGroups(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2dGroups, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, 
                              stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
# bottleneck type C
class BasicBlock(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, is_shortcut=False):
        super(BasicBlock, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2d(in_channel,    channel, kernel_size=3, padding=1, stride=stride)
        self.conv_bn2 = ConvBn2d(   channel,out_channel, kernel_size=3, padding=1, stride=1)

        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):
        z = F.relu(self.conv_bn1(x),inplace=True)
        z = self.conv_bn2(z)

        if self.is_shortcut:
            x = self.shortcut(x)

        z += x
        z = F.relu(z,inplace=True)
        return z

# Resnet34 classification
class ResNet34(nn.Module):
    def __init__(self, num_class=1000 ):
        super(ResNet34, self).__init__()

        self.block0  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1  = nn.Sequential(
             nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
             BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,),
          * [BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,) for i in range(1,3)],
        )
        self.block2  = nn.Sequential(
             BasicBlock( 64,128,128, stride=2, is_shortcut=True, ),
          * [BasicBlock(128,128,128, stride=1, is_shortcut=False,) for i in range(1,4)],
        )
        self.block3  = nn.Sequential(
             BasicBlock(128,256,256, stride=2, is_shortcut=True, ),
          * [BasicBlock(256,256,256, stride=1, is_shortcut=False,) for i in range(1,6)],
        )
        self.block4 = nn.Sequential(
             BasicBlock(256,512,512, stride=2, is_shortcut=True, ),
          * [BasicBlock(512,512,512, stride=1, is_shortcut=False,) for i in range(1,3)],
        )
        self.logit = nn.Linear(512,num_class)
        
    def forward(self, x):
        batch_size = len(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)
        return logit
    
# Resnet18 segmentation
class ResNet18(nn.Module):
    def __init__(self, num_class=1000 ):
        super(ResNet18, self).__init__()


        self.block0  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.block1  = nn.Sequential(
             nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
             BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,),
          * [BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,) for i in range(1,2)],
        )
        self.block2  = nn.Sequential(
             BasicBlock( 64,128,128, stride=2, is_shortcut=True, ),
          * [BasicBlock(128,128,128, stride=1, is_shortcut=False,) for i in range(1,2)],
        )
        self.block3  = nn.Sequential(
             BasicBlock(128,256,256, stride=2, is_shortcut=True, ),
          * [BasicBlock(256,256,256, stride=1, is_shortcut=False,) for i in range(1,2)],
        )
        self.block4 = nn.Sequential(
             BasicBlock(256,512,512, stride=2, is_shortcut=True, ),
          * [BasicBlock(512,512,512, stride=1, is_shortcut=False,) for i in range(1,2)],
        )
        self.logit = nn.Linear(512,num_class)

    def forward(self, x):
        batch_size = len(x)

        x = self.block0(x)
        x = F.max_pool2d(x,kernel_size=3, padding=1, stride=2, ceil_mode=False)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)
        return logit
    
class SENextBottleneckBlock(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, group=32, reduction=16, excite_size=-1, is_shortcut=False):
        super(SENextBottleneckBlock, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2dGroups(in_channel,     channel, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2dGroups(   channel,     channel, kernel_size=3, padding=1, stride=stride, groups=group)
        self.conv_bn3 = ConvBn2dGroups(   channel, out_channel, kernel_size=1, padding=0, stride=1)
        self.scale    = SqueezeExcite(out_channel, reduction, excite_size)

        if is_shortcut:
            self.shortcut = ConvBn2dGroups(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):
        z = F.relu(self.conv_bn1(x),inplace=True)
        z = F.relu(self.conv_bn2(z),inplace=True)
        z = self.scale(self.conv_bn3(z))

        if self.is_shortcut:
            z += self.shortcut(x)
        else:
            z += x

        z = F.relu(z,inplace=True)
        return z

class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, reduction=4, excite_size=-1):
        super(SqueezeExcite, self).__init__()
        self.excite_size=excite_size
        self.fc1 = nn.Conv2d(in_channel, in_channel//reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(in_channel//reduction, in_channel, kernel_size=1, padding=0)
        self.gather_excite = False

    def forward(self, x):

        if self.gather_excite:
            s = F.avg_pool2d(x, kernel_size=self.excite_size)
        else:
            s = F.adaptive_avg_pool2d(x,1)

        s = self.fc1(s)
        s = F.relu(s, inplace=True)
        s = self.fc2(s)

        if self.gather_excite:
            s = F.interpolate(s, size=(x.shape[2],x.shape[3]), mode='nearest')

        x = x*torch.sigmoid(s)
        return x
    
#resnext50_32x4d
class ResNext50(nn.Module):
    def __init__(self, num_class=1000 ):
        super(ResNext50, self).__init__()

        self.block0  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.block1  = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2, ceil_mode=True),
             SENextBottleneckBlock( 64, 128, 256, stride=1, is_shortcut=True,  excite_size=64),
          * [SENextBottleneckBlock(256, 128, 256, stride=1, is_shortcut=False, excite_size=64) for i in range(1,3)],
        )

        self.block2  = nn.Sequential(
             SENextBottleneckBlock(256, 256, 512, stride=2, is_shortcut=True,  excite_size=32),
          * [SENextBottleneckBlock(512, 256, 512, stride=1, is_shortcut=False, excite_size=32) for i in range(1,4)],
        )

        self.block3  = nn.Sequential(
             SENextBottleneckBlock( 512,512,1024, stride=2, is_shortcut=True,  excite_size=16),
          * [SENextBottleneckBlock(1024,512,1024, stride=1, is_shortcut=False, excite_size=16) for i in range(1,6)],
        )

        self.block4 = nn.Sequential(
             SENextBottleneckBlock(1024,1024,2048, stride=2, is_shortcut=True,  excite_size=8),
          * [SENextBottleneckBlock(2048,1024,2048, stride=1, is_shortcut=False, excite_size=8) for i in range(1,3)],
        )

        self.logit = nn.Linear(2048,num_class)


    def forward(self, x):
        batch_size = len(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)
        return logit


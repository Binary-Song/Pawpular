import torch.nn as nn 
import torch 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import albumentations as Alb
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np  
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
import torchvision.transforms.functional as functional


class PawpularDatasetLabelPreprocessor:

    def __init__(self): 
        self.bins = np.array([0,18,23,27,30,33,37,43,51,66,101])
   
    def transform(self, y, training=True):
        if training == True:
            return torch.from_numpy(np.digitize(y, self.bins) - 1) # 0-9 属于那一组
        else:
            return y
    
    def inverse_transform(self, y, training=True):
        y = y.cpu()
        y_l = self.bins[np.argmax(y,axis=1)]
        y_h = self.bins[np.argmax(y,axis=1) + 1]
        return torch.from_numpy((y_l + y_h)/2)

def conv(inplanes, outplanes, kernel_size=3, stride = 1, padding = 1, bias=False):
    return nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride,padding=padding, bias=bias)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv(inplanes, outplanes, stride=stride)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.activ = nn.SELU()
        self.conv2 = conv(outplanes, outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activ(out)

        return out


class MyNet(nn.Module):

    def __init__(
        self
    ):
        super().__init__()

        self.conv1 = conv(3, 64, kernel_size=7, padding=3, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.activ = nn.ELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res1 = self._make_layer(inplanes=64,outplanes=64, blocks=6,stride=1)
        self.res2 = self._make_layer(inplanes=64,outplanes=128, blocks=8,stride=2)
        self.res3 = self._make_layer(inplanes=128,outplanes=256, blocks=12,stride=2)
        self.res4 = self._make_layer(inplanes=256,outplanes=512, blocks=6,stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(86528, 256, False)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1, False) 

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.lastw = None

        #self.y_encoder = PawpularDatasetLabelPreprocessor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)   
        x = self.bn1(x)
        x = self.activ(x)
        x = self.maxpool(x)

        x = self.res1(x)
        x = self.activ(x)
        x = self.res2(x)
        x = self.activ(x)
        x = self.res3(x)
        x = self.activ(x)
        x = self.res4(x) 
        x = self.activ(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.activ(x)
        x = self.fc2(x) 

        x = self.sigmoid(x) * 100

        return x

    def get_loss_fn(self): 
        return nn.MSELoss()
    
    def _make_layer(self, inplanes, outplanes, blocks, stride=1):
        '''
        创建残差层
        inplanes: 输入的channel数量
        outplanes: 输出的channel数量
        blocks: 残差层主路Block数。
        '''
        downsample = None
        if inplanes != outplanes:
            # 此为短接采样层。当主路输出通道数或长宽与输入不同时，短路须用1*1卷积采样使维度一致
            downsample = nn.Sequential(
                conv(inplanes, outplanes, kernel_size=1,stride=stride,padding=0),
                nn.BatchNorm2d(outplanes)
                )
        layers = []
        layers.append(BasicBlock(inplanes, outplanes, stride, downsample))
        for _ in range(1,blocks):
            layers.append(BasicBlock(outplanes, outplanes ))
        return nn.Sequential(*layers)

    def _to_numpy(self, x):
        return x.cpu().detach().numpy()


class BaselineModel(nn.Module):

    def __init__(
        self,
        device,
        as_long = False,
        output_proba = False
    ):
        super().__init__() 
        self.device = device
        self.as_long = as_long
        self.output_proba = output_proba
        self.one_hot = OneHotEncoder(sparse=False)

    def fit(self, y_train):
        self.mean = torch.mean(torch.from_numpy(y_train).to(self.device))
        self.one_hot.fit(y_train)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        x = torch.ones(x.shape[0],1).to(self.device) * self.mean

        if self.as_long == True: 
            x = x.long()

        if self.output_proba == True:
            x = x.long()
            x = self.one_hot.transform(x.cpu())
            x = torch.from_numpy(x).to(self.device)
        return x


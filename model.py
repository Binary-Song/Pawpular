import torch.nn as nn
import pretrainedmodels 
import torch 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import albumentations as Alb
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np  
from torch.utils.data import DataLoader

class PawpularDatasetImagePreprocessor:
    
    def __init__(self):
        pass

    def partial_fit(self, X): 
        pass
    
    def fit(self, X):
        pass

    def transform(self, X): 
        T = tv.transforms.Compose([ 
            tv.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        t = T(X)
        # n = t.cpu().numpy()
        # plt.imshow(np.transpose(n[0], (1,2,0)))
        # plt.imshow(np.transpose(X[0], (1,2,0)))
        # plt.show()
        return t  

class PawpularDatasetLabelPreprocessor:

    def __init__(self): 
        self.bins = np.array([0,18,23,27,30,33,37,43,51,66,101])
   
    def transform(self, y):
        return torch.from_numpy(np.digitize(y, self.bins) - 1) # 0-9 属于那一组
    
    def inverse_transform(self, y):
        y = y.cpu()
        y_l = self.bins[np.argmax(y,axis=1)]
        y_h = self.bins[np.argmax(y,axis=1) + 1]
        return torch.from_numpy((y_l + y_h)/2)

def conv(inplanes, outplanes, kernel_size=3, stride = 1, padding = 1):
    return nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride,padding=padding, bias=False)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv(inplanes, outplanes, stride=stride)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.activ = nn.SELU(inplace=True)
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


class ResNet(nn.Module):

    def __init__(
        self, 
        layers  
    ): 
        super().__init__()
        inplanes = 64
        self.conv1 = conv(3, inplanes,kernel_size=7,padding=3)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.activ = nn.SELU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(inplanes, 64, layers[0])
        self.layer2 = self._make_layer(64,  128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2) 
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(131072, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10) 
        self.softmax = nn.Softmax()
        self.y_proc = PawpularDatasetLabelPreprocessor()

        # init args

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules(): 
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activ(x) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)

        if self.training == False: # if in eval mode
            x = self.y_proc.inverse_transform(x).cuda() # return popularity score
        # else return which bin score is in (vector)
        return x

    def get_loss_fn(self):
        if self.training == True: # train mode
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _make_layer(self, inplanes, outplanes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                conv(inplanes, outplanes,kernel_size=1,stride=stride,padding=0),
                nn.BatchNorm2d(outplanes)
                )
        layers = []
        layers.append(BasicBlock(inplanes, outplanes, stride, downsample))
        for _ in range(1,blocks):
            layers.append(BasicBlock(outplanes, outplanes ))
        return nn.Sequential(*layers)




class BaselineModel(nn.Module):

    def __init__(
        self, 
        device
    ):
        super().__init__() 
        self.device = device

    def fit(self, y_train):
        self.mean = torch.mean(torch.from_numpy(y_train).to(self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return torch.ones(x.shape[0],1).to(self.device) * self.mean

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
from sklearn.preprocessing import OneHotEncoder

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

def conv(inplanes, outplanes, kernel_size=3, stride = 1, padding = 1, bias=False):
    return nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride,padding=padding, bias=bias)

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
    
        self.activ = nn.ReLU()
    
        self.conv1 = conv(1,  64, kernel_size=3, padding=1,stride=1,bias=True) 
        self.conv2 = conv(64, 128, kernel_size=3, padding=1,stride=1, bias=True) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(14*14*128,1024) 
        self.fc2 = nn.Linear(1024, 10) 

        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)   
        x = self.activ(x)
        x = self.conv2(x)
        x = self.activ(x)
        x = self.pool(x) 

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.activ(x) 
        x = self.dropout(x)
        x = self.fc2(x) 

        x = self.softmax(x)
 
        return x

    def get_loss_fn(self): 
        return nn.CrossEntropyLoss()
      

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
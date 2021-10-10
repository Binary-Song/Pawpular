import torch.nn as nn
import torch 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import albumentations as A
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np  
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder

class MyAug:
    
    def __init__(self, resize, crop_size):
        self.crop_size = crop_size
        self.resize = resize
    
        self.T = A.Compose([
            A.Resize(height=resize, width=resize, always_apply=True),
            A.RandomCrop(width=crop_size, height=crop_size),
            A.HorizontalFlip(p=0.5)
        ]) 
    
    def _show_tensor_hwc(self, X):
        plt.imshow(X)

    def _show_tensor_chw(self, X): 
        plt.imshow(np.transpose(X, (1,2,0)))

    def transform(self, X, training=True): 
        # self._show_tensor_chw(X)
        X = np.transpose(X, (1,2,0)) # C H W -> H W C
        X = self.T(image=X)['image']
        # self._show_tensor_hwc(X)
        X = np.transpose(X, (2,0,1)) # H W C -> C H W
        return X
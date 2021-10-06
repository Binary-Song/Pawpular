import os
import pandas as pd
import numpy as np
import albumentations as A
import torch
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold

from engine import do_epochs
from ImageDataset import ImageDataset
from model import *
from torch.utils.data import DataLoader
from ImageSource import ImageSource

import torchvision
import torch
from torchvision import datasets, transforms

from metrics import * 

train_img_dir = "./data/train"

if True:
    train_csv = "./data/train.csv"
else:
    train_csv = "./data/train_small.csv"

cache_path = "./data/cache/image_cache.pickle.bin"

def get_img_paths(df): 
    paths = []
    for fname in df["Id"]:
        path = f"{train_img_dir}/{fname}.jpg"
        paths.append(path)
    return paths

def get_targets(df):
    targets = df["Pawpularity"].to_numpy(dtype=np.float64) 
    return targets

def make_cache():
    df = pd.read_csv(train_csv)
    paths = get_img_paths(df)
    src = ImageSource(resize=(128,128))
    src.make_cache(paths, output_path=cache_path)

if __name__ == '__main__':

    MAKE_CACHE = False

    if MAKE_CACHE == True:
        make_cache()
        exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    else:
        raise Exception("Use gpu to train instead!")

    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, ],std=[0.5, ])])

    df = pd.read_csv(train_csv).sample(frac=1.0,random_state=42)

    paths = get_img_paths(df) # array
    targets = get_targets(df) # numpy

    src = ImageSource(resize=(128,128))
    src.load_cache(cache_path)

    model = ResNet().to(device)  

    baseline_model = BaselineModel(device, as_long=False, output_proba=False)
 

    fold_maker = KFold(n_splits=5, shuffle=True, random_state=42)
    for k, (train_idx, valid_idx) in enumerate(
        fold_maker.split(X=paths, y=targets)):

        print(f"fold {k}")

        img_size = (224, 224)
        batch_size = 32
        num_workers = 0

        train_set = ImageDataset(
            paths=[paths[i] for i in train_idx],
            src=src,
            targets=targets[train_idx],
            resize=img_size)

        valid_set = ImageDataset(
            paths=[paths[i] for i in valid_idx],
            src=src,
            targets=targets[valid_idx],
            resize=img_size)

        train_loader = DataLoader(
            train_set, batch_size=batch_size, num_workers=num_workers)

        valid_loader = DataLoader(
            valid_set, batch_size=batch_size, num_workers=num_workers)
  
        optim = torch.optim.Adam(model.parameters()) 

        do_epochs(
            epochs=1000, 
            train_loader=train_loader, 
            valid_loader=valid_loader, 
            model=model,
            baseline_model=baseline_model,
            optimizer=optim,  
            metrics=[ 
                    R2Score(),
                    NegMeanAbsError()
                    ], 
            X_prepro=PawpularDatasetImagePreprocessor(),
            y_prepro=PawpularDatasetLabelPreprocessor(),
            device=device) 
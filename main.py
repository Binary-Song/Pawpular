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

    data_train = datasets.MNIST(root = "./data/",
                                transform=transform,
                                train = True,
                                download = True)
    
    data_test = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = False,
                            download = True)
                                
    train_loader = DataLoader(
                            dataset=data_train,
                            batch_size = 64,
                            shuffle=True)
    
    valid_loader = DataLoader(
                            dataset=data_test,
                            batch_size = 64,
                            shuffle=True)

    model = ResNet([1,1,1,1]).to(device)  

    baseline_model = BaselineModel(device, as_long=True, output_proba=True)

    optim = torch.optim.Adam(model.parameters()) 

    do_epochs(
        epochs=1000, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        model=model,
        baseline_model=baseline_model,
        optimizer=optim,  
        metrics=[
                AccuracyScore(),
                F1Score(average='weighted'),
                RocAucScore(average='macro', multi_class='ovr'), 
                RocAucScore(average='weighted', multi_class='ovr'),], 
        X_prepro=None, 
        y_prepro=None, 
        device=device)

        # train_plans = [{"name": "baseline", 
        #                 "model": baseline, 
        #                 "epochs": 1,
        #                 "optim": None},

        #                {"name": "model", 
        #                 "model": model, 
        #                 "epochs": epochs,
        #                 "optim": optim}]

        # for plan in train_plans:

        #     name = plan['name']
        #     model = plan['model']
        #     epochs = plan['epochs']
        #     optim = plan['optim']

        #     print(f"training: {name}")

        #     for epoch in range(epochs): 
        #         train_loss = engine.train(
        #             train_loader,
        #             model, 
        #             optimizer=optim, 
        #             loss_fn=nn.CrossEntropyLoss(),
        #             device=device, 
        #             X_preprocessor=X_prep,
        #             y_preprocessor=y_prep)

        #         output, targets = engine.evaluate(
        #             valid_loader,
        #             model, 
        #             device=device, 
        #             X_preprocessor=X_prep, 
        #             y_preprocessor=y_prep)

        #         valid_r2 = metrics.r2_score(targets, output)
        #         valid_error = metrics.mean_absolute_error(targets, output)

        #         print( f"epoch {epoch}/{epochs}, train loss {train_loss}, valid R-square {valid_r2}, valid error {valid_error}")

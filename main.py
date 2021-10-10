import os
import pandas as pd
import numpy as np
import albumentations as A
import torch
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from colorama import Fore, Back, Style 
from engine import do_epochs
from ImageDataset import ImageDataset
from model import MyNet, BaselineModel
from torch.utils.data import DataLoader
from ImageSource import ImageSource
from preprocessor import MyAug
import torchvision
import torch
import sys
from torchvision import datasets, transforms

from metrics import *

train_img_dir = "./data/train"
save_dir = "./log/checkpoints/"

if len(sys.argv) >= 2 and sys.argv[1] == '-d':
    train_csv = "./data/train_small.csv"
    use_trimmed_data = True
else:
    train_csv = "./data/train.csv"
    use_trimmed_data = False

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

def make_cache(resize):
    if os.path.isfile(cache_path):
        ans = input('cache exists, override? (y/n)')
        if ans == 'y':
            pass 
        else:
            exit()
    df = pd.read_csv(train_csv)
    paths = get_img_paths(df)
    src = ImageSource()
    src.make_cache(paths, output_path=cache_path, resize=resize)

if __name__ == '__main__':

    if use_trimmed_data:
        print(f"{Fore.RED}{Back.WHITE} warning: using trimmed data {Back.RESET}{Fore.RESET}")

    MAKE_CACHE = False

    if MAKE_CACHE == True:
        make_cache(resize=(256,256))
        exit()

    resize = 220
    crop_size = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],std=[0.5])])

    df = pd.read_csv(train_csv).sample(frac=1.0,random_state=42)

    paths = get_img_paths(df) # array
    targets = get_targets(df) # numpy

    src = ImageSource()
    src.load_cache(cache_path)

    model = MyNet().to(device)

    baseline_model = BaselineModel(device, as_long=False, output_proba=False)

    fold_maker = KFold(n_splits=5, shuffle=True, random_state=42)
    for k, (train_idx, valid_idx) in enumerate(
        fold_maker.split(X=paths, y=targets)):

        print(f"-- fold {k} --")

        batch_size = 24
        num_workers = 0

        train_set = ImageDataset(
            paths=[paths[i] for i in train_idx],
            src=src,
            targets=targets[train_idx],
            resize=resize,
            crop_size=crop_size)

        valid_set = ImageDataset(
            paths=[paths[i] for i in valid_idx],
            src=src,
            targets=targets[valid_idx],
            resize=resize,
            crop_size=crop_size)

        train_loader = DataLoader(
            train_set, batch_size=batch_size, num_workers=num_workers)

        valid_loader = DataLoader(
            valid_set, batch_size=batch_size, num_workers=num_workers)

        optim = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=35, eta_min=0.00001)
        do_epochs(
            epochs=1000,
            train_loader=train_loader, 
            valid_loader=valid_loader, 
            model=model,
            baseline_model=baseline_model,
            optimizer=optim,
            scheduler=scheduler,  
            metrics=[R2Score(), NegMeanAbsError(), ExplainedVariance(), NegMaxError()], 
            X_prepro=None,
            y_prepro=None,
            device=device,
            save_dir=save_dir,
            #load_path='./log/checkpoints/20211008T175343/params_at_epoch_25.bin'
            )

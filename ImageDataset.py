import torch
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageFile
from skimage import io
import skimage.transform as tr
import matplotlib.pyplot as plt
from tqdm import tqdm
from ImageSource import ImageSource 
from preprocessor import MyAug
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset:

    def __init__(self, paths, src: ImageSource, targets, resize=None, crop_size=None):
        self.src = src 
        self.paths = paths
        self.targets = targets
        self.preprocess = MyAug(resize, crop_size)
  
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        
        img = self.src[self.paths[i]]

        img = self.preprocess.transform(img)

        y = self.targets[i]

        return img, y

# if __name__ == "__main__":
    # train_img_dir = "./data/train"
    # train_csv = "./data/train.csv"
    # df = pd.read_csv(train_csv)
    # file_names = []
    # for fname in df["Id"]:
    #     path = f"{train_img_dir}/{fname}.jpg"
    #     file_names.append(path)
    # dataset = ImageDataset(
    #     file_names, df["Pawpularity"], 'r', resize=(128, 128))
    # data = dataset.__getitem__(485)
    # img = data["image"]
    # plt.imshow(img)
    # plt.show()

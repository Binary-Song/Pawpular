import pickle
import skimage.transform as imgt
import skimage.io as io
import numpy as np
import math
from tqdm import tqdm
 
class ImageSource:
    
    def __init__(self, cache_max_size = math.inf):
        '''
        图像集合，负责从缓存或硬盘中读取图像，亦可将缓存写入硬盘。
        '''
        self.cache = {}
        self.cache_max_size = cache_max_size

    def make_cache(self, paths, output_path, resize = None): 
        '''
        制作缓存
        paths ([str]): 图像的路径
        output_path (str): 图像的输出路径
        resize ((int,int)): 若不为None，则将图像缩放后保存
        '''
        for path in tqdm(paths, desc='making cache'):
            self.cache[path] = self.read_image(path, resize = resize)
 
        with open(output_path, 'wb') as file:
            pickle.dump(self.cache, file)
    
    def load_cache(self, input_path):
        '''
        加载缓存
        input_path (str): 缓存文件路径 
        '''
        with open(input_path,'rb') as file:
            self.cache = pickle.load(file)

    def read_image(self, path, resize = None):
        '''
        从文件系统中读取图像 
        path (str): 路径
        resize ((int,int)): 若不为None，则将图像缩放
        返回 (numpy): 图像张量。形状：(C,H,W)，像素值用0-1内的float32表示。
        '''
        img = io.imread(path) 
        if resize is not None:
            img = imgt.resize(img, resize) 
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) # H W C -> C H W
        return img

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, path):
        if path in self.cache: 
            return self.cache[path]
        else:
            img = self.read_image(path) 
            if len(self.cache) < self.cache_max_size:
                self.cache[path] = img
            return img
    
    
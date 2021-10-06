import pickle
import skimage.transform as imgt
import skimage.io as io
import numpy as np
import math
from tqdm import tqdm
 
class ImageSource:
    
    def __init__(self, resize=None, transpose_for_torch = True, cache_max_size = math.inf): 
        self.resize = resize
        self.transpose_for_torch = transpose_for_torch
        self.cache = {}
        self.cache_max_size = cache_max_size

    def make_cache(self, paths, output_path = None): 
        for path in tqdm(paths, desc='making cache'):
            self.cache[path] = self.read_image(path)
        if output_path is not None:
            with open(output_path, 'wb') as file:
                pickle.dump(self.cache, file)
    
    def load_cache(self, input_path):
        with open(input_path,'rb') as file:
            self.cache = pickle.load(file)

    def read_image(self, path):
        img = io.imread(path) 
        if self.resize is not None:
            img = imgt.resize(img, self.resize) 
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
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
    
    
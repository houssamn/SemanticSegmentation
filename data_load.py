import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2


class MaskDataset(Dataset):
    """
        A collection of input images and corresponding masks. 
    """
    def __init__(self, path_to_df, transform=None): 
        """
            path_to_df : a dataframe containing two columns 'SavedMaskPath' & 'SavedImagePath'
        """
        self.df = pd.read_csv(path_to_df)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx]['SavedImagePath']) 
        mask = cv2.imread(self.df.iloc[idx]['SavedMaskPath']) 
        
        obj = {"image": image, "mask": mask}

        if(self.transform):
            obj = self.transform(obj)
        
        return obj
    
    
class Normalize(object):
    """Normalize the pixel values in each channel to the range [0,1]."""        

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        image_copy = np.copy(image)
        mask_copy = np.copy(mask)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
        mask_copy = mask_copy/255.0    
        
        return {'image': image_copy, 'mask': mask_copy}

class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {"image": torch.from_numpy(image), "mask":torch.from_numpy(mask)}


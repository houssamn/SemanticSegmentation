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
        mask = np.round(mask[:, :, [0]]) # Select only 1 channel from the mask
        
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


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        msk = cv2.resize(mask, (new_w, new_h))
        
        # Add the dummy channel dimension to the mask
        if(len(mask.shape) == 3 and len(msk.shape) == 2):
            msk = msk[:, :, np.newaxis]

        return {'image': img, 'mask': msk}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        mask = mask[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'mask': mask}

class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        return {"image": torch.from_numpy(image).float(), "mask":torch.from_numpy(mask).float()}

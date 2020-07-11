import os
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import io, transform

import torch
import torchvision






def missing_keys(folder_path):
    missing_values = []
    keys = {}
    value = 885
    for i in range(13,886):
        path = os.path.join(folder_path, f'img_{str(i)}.jpg')
        try:
            Image.open(path)
        except:
            missing_values.append(i)
            if i!=863:
                keys[i] = value
                value = value-1
            else:
                keys[i] = 862
        finally:
            pass
    return missing_values, keys


class train_dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, channel='RGB'):
        """
        Args:
            root_dir (string): Directory with images
            transform (callable, optional): Optional transform to be applied
                on a sample
            channels (string): ['RGB'(default), 'CMYK'] color channels to convert
                all images to

        Note: Fig.0 = Fig.819 for labels and captions!!!!
        """

        self.root_dir = root_dir
        self.transform = transform
        self.channels = channel
        self.num_figs = 819 + 850

    def __len__(self):
        return self.num_figs

    def __getitem__(self, index):
        
        if index<819:
            img_path = os.path.join(self.root_dir, f'Fig.{index}.jpg')
            print(img_path)
        else:
            missing_values, key_dict = missing_keys(self.root_dir)
            if index-806 in missing_values:
                img_path = os.path.join(self.root_dir, f'img_{key_dict[index-806]}.jpg')
                print(img_path)
            else:
                img_path = os.path.join(self.root_dir, f'img_{index-806}.jpg')
                print(img_path)
    
        image = Image.open(img_path)
        #--------------------------------------------------------------------------------
        if self.channels=='RGB':
            if image.mode=='CMYK':
                image = image.convert('RGB')
        elif self.channels=='CMYK':
            if image.mode=='RGB':
                image = image.convert('CMYK')
        else:
            print("Invalid Channel Type")

        #---------------------------------------------------------------------------------
        
        if self.transform:
            image = self.transform(image)
        sample = {'image': image}
        
        return sample
    

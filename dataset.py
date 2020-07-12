import os
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import io, transform

import torch
import torchvision






def missing_keys(folder_path):
    #missing_values = []
    keys = {}
    value = 885
    for i in range(13,886):
        path = os.path.join(folder_path, f'part1/Images/img_{str(i)}.jpg')
        try:
            Image.open(path)
        except:
            #missing_values.append(i)
            if i!=863:
                keys[i] = value
                value = value-1
            else:
                keys[i] = 862
        finally:
            pass
    return keys


class train_dataset(torch.utils.data.Dataset):
    def __init__(self, train_dict_1, root_dir, transform=None, channel='RGB'):
        """
        Args:
            root_dir (string): Train Directory with images
            train_dict_1 (dictionary): Dictionary containing keys corresponding to missing indexes
                in the first train image folder and the alternate values to be assigned to these indexes
            transform (callable, optional): Optional transform to be applied
                on a sample
            channels (string): ['RGB'(default), 'CMYK'] color channels to convert
                all images to
        """

        self.root_dir = root_dir
        self.key_dict = train_dict_1
        self.missing_values = train_dict_1.keys()
        self.transform = transform
        self.channels = channel
        self.num_figs = 819 + 850

    def __len__(self):
        return self.num_figs

    def __getitem__(self, index):
        
        if index==0:
            img_path = os.path.join(self.root_dir, f'part1/Images/Fig.{index+819}.jpg')
        elif index<819:
            img_path = os.path.join(self.root_dir, f'part1/Images/Fig.{index}.jpg')
            print(img_path)
        elif index<1670:
            #missing_values, key_dict = missing_keys(self.root_dir)
            if index-806 in self.missing_values:
                img_path = os.path.join(self.root_dir, f'part1/Images/img_{self.key_dict[index-806]}.jpg')
                print(img_path)
            else:
                img_path = os.path.join(self.root_dir, f'part1/Images/img_{index-806}.jpg')
                print(img_path)
        #--------------------------------------------------------------------------------
        
        #else:
        #img_path = os.path.join(
        
        #--------------------------------------------------------------------------------


    
        image = Image.open(img_path)
                                    
        if self.channels=='RGB':
            if image.mode=='CMYK':
                image = image.convert('RGB')
        elif self.channels=='CMYK':
            if image.mode=='RGB':
                image = image.convert('CMYK')
        else:
            print("Invalid Channel Type")

        
        if self.transform:
            image = self.transform(image)
        sample = {'image': image}
        
        return sample
    

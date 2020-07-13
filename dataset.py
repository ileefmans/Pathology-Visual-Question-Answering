import os
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import io, transform

import torch
import torchvision




# Function to generate dictionary for path to second train folder
def path_dict(folder_path):
    dic = {}
    for i in range(3329):
        for j in range(78):
            path = os.path.join(folder_path ,f'part2/part2_images/{str(j)}', f'{str(i)}.jpg')
            try:
                image = Image.open(path)
                dic[i] = j
            except:
                pass
            finally:
                pass

    return dic



# Class to create dataset with annotation
class train_dataset(torch.utils.data.Dataset):
    def __init__(self, annotation_dir, train_dict, img_dir, transform=None, channel='RGB'):
        """
            Args:
            img_dir (string): Train Directory with images
            
            train_dict (dictionary): Dictionary containing folder and image matches for second part of 
                train data
                
            transform (callable, optional): Optional transform to be applied
            on a sample
            
            channels (string): ['RGB'(default), 'CMYK'] color channels to convert
                all images to
            """
        
        self.img_dir = img_dir
        self.annotation = pd.read_json(annotation_dir)
        self.new_annotation = self.annotation.loc[self.annotation.Images!='img_233',:].reset_index()
        self.train_dict2 = train_dict
        self.transform = transform
        self.channels = channel
    
    def __len__(self):
        return len(self.new_annotation)
    
    def __getitem__(self, index):
        
        image_name = self.new_annotation.Images[index]

        if image_name[0] in ['F','i']:
            img_path = os.path.join(self.img_dir, f'part1/Images/{image_name}.jpg')
            print(img_path)
        else:
            img_path = os.path.join(self.img_dir, f'part2/part2_images/{self.train_dict2[int(image_name[:-4])]}/{image_name}')
            print(img_path)
    
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



        question = self.new_annotation.Questions[index]
        answer = self.new_annotation.Answers[index]
        sample = {'image': image, 'question': question, answer: 'answer'}
    
        return sample




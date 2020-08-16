import os
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import io, transform

import torch
from torch import nn
import torchvision

from preprocess import image_process





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



# Generalizing the above functions

def path_dict(folder_path, total_pics, total_folders):
    dic = {}
    for i in range(total_pics):
        for j in range(total_folders):
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
    def __init__(self, annotation_dir, train_dict, img_dir, transform=None, img_size=(491, 600)):
        """
            Args:
            
            annotation_dir (string): Directory too json containg training annotation
            
            train_dict (dictionary): Dictionary containing folder and image matches for second part of
                train data
            
            img_dir (string): Train Directory with images
                
            transform (callable, optional): Optional transform to be applied
                on a sample
            
            img_size (tuple): (height, width) Desired height and width for all images to conform to. Height
                must equal width.
            """
        
        self.img_dir = img_dir
        self.annotation = pd.read_json(annotation_dir)
        self.new_annotation = self.annotation.loc[self.annotation.Images!='img_233',:].reset_index()
        self.train_dict2 = train_dict
        self.transform = transform
        self.img_size = img_size
        self.preprocess = image_process(self.img_size)

    
    def __len__(self):
        return len(self.new_annotation)
    
    def __getitem__(self, index):
        
        image_name = self.new_annotation.Images[index]

        if image_name[0] in ['F','i']:
            img_path = os.path.join(self.img_dir, f'part1/Images/{image_name}.jpg')
            #print(img_path)
        else:
            img_path = os.path.join(self.img_dir, f'part2/part2_images/{self.train_dict2[int(image_name[:-4])]}/{image_name}')
            #print(img_path)
    
        image = Image.open(img_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        

        image = self.preprocess.expand(image)
        


        if self.transform:
            image = self.transform(image)
        
        image = self.preprocess.uniform_size(image)
        
        
        
        question = self.new_annotation.Questions[index]
        #print(question)
        #print(type(question))
        #cln_txt= text_process(question)
        #question = cln_txt.text_preprocess()
        answer = self.new_annotation.Answers[index]
        sample = {'image': image, 'question': question, 'answer': answer}
    
        return sample




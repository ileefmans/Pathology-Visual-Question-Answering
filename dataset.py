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


# Function to expand or reduce size of image keeping aspect ration intact
def expand(image, largest_dimension, resample=0):
    #resample options (see Pil.Image.resize for more documentation):
    #0 :Nearest neighbors (default)
    #PIL.Image.BILINEAR
    #Image.BICUBIC
    #Image.LANCZOS
    if image.height>= image.width:
        new_width = round((image.width/image.height)*largest_dimension)
        new_height = largest_dimesion
    else:
        new_height = round((image.height/image.width)*largest_dimension)
        new_width = largest_dimension
    image = image.resize((new_width, new_height), resample=resample )
    return image

# Function to add padding to make pictures uniform size
def uniform_size(x, height, width):
    tup_val1 = round(((width-x.size()[2])/2)-.1)
    tup_val2 = round(((width-x.size()[2])/2)+.1)
    tup_val3 = round(((height-x.size()[1])/2)-.1)
    tup_val4 = round(((height-x.size()[1])/2)+.1)
    
    tup = (tup_val1, tup_val2, tup_val3, tup_val4)
    
    pad = nn.ZeroPad2d(tup)
    padded = pad(x)
    return padded



# Class to create dataset with annotation
class train_dataset(torch.utils.data.Dataset):
    def __init__(self, annotation_dir, train_dict, img_dir, transform=None, img_size=(600,600)):
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
        #self.channels = channel
        self.img_size = img_size
    
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
        
        #if self.channels=='RGB':
        if image.mode=='CMYK':
            image = image.convert('RGB')
        #elif self.channels=='CMYK':
            #if image.mode=='RGB':
                #image = image.convert('CMYK')
        else:
            print("Invalid Channel Type")

        if img_size[0] == img_size[1]:
            image = expand(image, img_size[0])
        else:
            raise Exception("Attribute Error: Image size must be square")
        


        if self.transform:
            image = self.transform(image)
        
        image = uniform_size(image, img_size[0], img_size[1])
        



        question = self.new_annotation.Questions[index]
        answer = self.new_annotation.Answers[index]
        sample = {'image': image, 'question': question, 'answer': answer}
    
        return sample




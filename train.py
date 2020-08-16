import os
from PIL import Image
import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt

import torch
from torch import nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
import torchvision

from dataset import train_dataset, path_dict

from preprocess import image_process

from nlp_preprocess import text_process

from pretrain_VAE import VAE



def get_args():
    parser = argparse.ArgumentParser(description = "Model Options")
    parser.add_argument("--annotation_path", type=str, default="/Users/ianleefmans/Desktop/data/train/part1/part1_2.json", help="Path to annotation .json")
    parser.add_argument("--train_img_path", type=str, default="/Users/ianleefmans/Desktop/data/train", help="Path to training image folder")
    parser.add_argument("--batch_size", type=int, default=4, help="Mini-batch size")
    parser.add_argument("--number_of_channels", type=int, default=16, help="Number of channels to map to in first layer")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for Adam Optimizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs model will be trained for")

    return parser.parse_args()



class Trainer:
    def __init__(self):
        
        # initialize arguments
        self.ops = get_args()
        self.device = "cuda"
        self.annotation_path = self.ops.annotation_path
        self.train_img_path = self.ops.train_img_path
        self.index_dict = path_dict(self.train_img_path)
        self.transform = torchvision.transforms.ToTensor()
        self.batch_size = self.ops.batch_size
        self.num_channels = self.ops.number_of_channels
        self.learning_rate = self.ops.learning_rate
        self.height = 491
        self.wdith = 600
        self.epochs = self.ops.epochs
        
        
        # initialize dataloader
        self.train_set = train_dataset(self.annotation_path, self.index_dict,
                                       img_dir = self.train_img_path,
                                       transform = self.transform)
        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True)


        #initialize models
        self.vae = VAE(self.num_channels).to(self.device)
        
        
        #initialize optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
    
    
    # Loss Function
    def loss_fcn(self, x_hat, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(x_hat.view(-1, 3*self.height*self.width), x.view(-1, 3*self.height*self.width),  reduction='sum')
        KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
        return BCE+KLD


    def testing(self):
        sample = iter(self.dataloader).next()
        print(sample['image'][0].size())
        print(sample['image'].size())
        plt.imshow(torchvision.transforms.ToPILImage(mode='RGB')(sample['image'][0]))
        pass


    def train(self):
        codes = dict(mulis=list(), logsig2=list(), y=list())
        for epoch in range(0, self.epochs+1):
            if epoch>0:
                self.vae.train()
                train_loss=0
                for x in self.train_loader:












if __name__ == "__main__":
    trainer = Trainer()
    trainer.testing()









import os
from PIL import Image
import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt
from IPython import display
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
import torchvision

from dataset import create_dataset, path_dict

from preprocess import image_process

from nlp_preprocess import text_process

from pretrain_VAE import VAE



def get_args():
    parser = argparse.ArgumentParser(description = "Model Options")
    parser.add_argument("--train_annotation_path", type=str, default="/Users/ianleefmans/Desktop/data/train/part1/part1_2.json", help="Path to train annotation .json")
    parser.add_argument("--train_img_path", type=str, default="/Users/ianleefmans/Desktop/data/train", help="Path to training image folder")
    parser.add_argument("--val_annotation_path", type=str, default="/Users/ianleefmans/Desktop/data/val/part3.json", help="Path to train annotation .json")
    parser.add_argument("--val_img_path", type=str, default="/Users/ianleefmans/Desktop/data/val", help="Path to validation image folder")
    parser.add_argument("--batch_size", type=int, default=4, help="Mini-batch size")
    parser.add_argument("--number_of_channels", type=int, default=16, help="Number of channels to map to in first layer")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for Adam Optimizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs model will be trained for")

    return parser.parse_args()



class Trainer:
    def __init__(self):
        
        # Initialize Arguments
        self.ops = get_args()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_annotation_path = self.ops.train_annotation_path
        self.train_img_path = self.ops.train_img_path
        self.val_annotation_path = self.ops.val_annotation_path
        self.val_img_path = self.ops.val_img_path
        self.train_index_dict = path_dict(self.train_img_path, 3329, 78)
        self.val_index_dict = path_dict(self.val_img_path, 2424, 25, training=False)
        self.transform = torchvision.transforms.ToTensor()
        self.batch_size = self.ops.batch_size
        self.num_channels = self.ops.number_of_channels
        self.learning_rate = self.ops.learning_rate
        self.height = 491
        self.width = 600
        self.epochs = self.ops.epochs
        
        

        # Preprocess Questions and Answers
        
        self.train_annotation = pd.read_json(self.train_annotation_path)
        self.train_annotation = self.train_annotation.loc[~self.train_annotation.Images.isin(['img_233']),:].reset_index()

        self.clean_train_ques = text_process(self.train_annotation.Questions)
        self.train_questions = torch.tensor(self.clean_train_ques.text_preprocess()[0])
        

        self.clean_train_ans = text_process(self.train_annotation.Answers)
        self.train_answers = torch.tensor(self.clean_train_ans.text_preprocess()[0])
        
        
        
        self.val_list = [615, 657, 992, 1001, 1237, 1247, 1260, 1419, 1705, 1996, 2237]
        
        self.val_annotation = pd.read_json(self.val_annotation_path)
        self.val_annotation = self.val_annotation.loc[~self.val_annotation.Images.isin(self.val_list),:].reset_index()
        
        self.clean_val_ques = text_process(self.val_annotation.Questions)
        self.val_questions = torch.tensor(self.clean_val_ques.text_preprocess()[0])
        
        self.clean_val_ans = text_process(self.val_annotation.Answers)
        self.val_answers = torch.tensor(self.clean_val_ans.text_preprocess()[0])
        
  
        
        # Initialize Dataloaders
        self.train_set = create_dataset(self.train_annotation_path, self.train_questions, self.train_answers,
                                        self.train_index_dict, img_dir = self.train_img_path,
                                        transform = self.transform, training=True)
        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size, num_workers=self.batch_size, shuffle=True)
        
        
        self.val_set = create_dataset(self.val_annotation_path, self.val_questions, self.val_answers,
                                      self.val_index_dict, img_dir = self.val_img_path,
                                      transform = self.transform, training=False)
        self.val_loader = DataLoader(dataset=self.val_set, batch_size=self.batch_size, num_workers = self.batch_size, shuffle=True)
        
        


        #Initialize Models
        self.vae = VAE(self.num_channels).to(self.device)
        
        
        #Initialize Optimizer
        self.optim = torch.optim.Adam(self.vae.parameters(), lr = self.learning_rate)
    
    
    # Define Loss Function
    def loss_fcn(self, x_hat, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(x_hat.view(-1, 3*self.height*self.width), x.view(-1, 3*self.height*self.width),  reduction='sum')
        KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
        return BCE+KLD

    # Functions to Test if Data is Loaded Correctly
    def training(self):
        sample = iter(self.dataloader).next()
        print(sample['image'][0].size())
        print(sample['image'].size())
        plt.imshow(torchvision.transforms.ToPILImage(mode='RGB')(sample['image'][0]))
        pass
    
    def testing(self):
        
        sample = iter(self.val_loader).next()
        print(sample['image'][0].size())
        print(sample['image'].size())
        plt.imshow(torchvision.transforms.ToPILImage(mode='RGB')(sample['image'][0]))
        pass



    # TRAINING
    def train(self):
        
        print("\n \n \n Training and Validaation Results: \n \n")
        
        codes = dict(mulis=list(), logsig2=list(), y=list())
   
        for epoch in range(0, self.epochs+1):
            
            
            #Training
            if epoch>0:
                self.vae.train()
                train_loss=0

                for x, question, answer in tqdm(self.train_loader, desc= "Train Epoch "+str(epoch)):
                    x = x.to(self.device)
                    x_hat, mu, logvar = self.vae(x)
                    loss = self.loss_fcn(x_hat, x, mu, logvar)
                    train_loss += loss
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                
                print(f'====> Epoch: {epoch} Average loss: {train_loss / len(self.train_loader.dataset):.4f}')


            #Testing
            means, logvars, labels = list(), list(), list()

            with torch.no_grad():
                self.vae.eval()
                test_loss=0
                
                for x, question, answer in tqdm(self.val_loader, desc="Val Epoch "+str(epoch)):
                    x = x.to(self.device)
                    x_hat, mu, logvar = self.vae(x)
                    test_loss += self.loss_fcn(x_hat, x, mu, logvar).item()
                    means.append(mu.detach())
                    logvars.append(logvar.detach())

            codes['mulis'].append(torch.cat(means))
            codes['logsig2'].append(torch.cat(logvars))
            test_loss /= len(self.val_loader.dataset)
            print(f'====> Test set loss: {test_loss:.4f}')








if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()









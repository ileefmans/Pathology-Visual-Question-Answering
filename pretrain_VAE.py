import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision

import os

# takeout part where tokens=list(tok.word_count)
# use doclist instead because tokenizer already makes tokens obviously


class Flatten(nn.Module):
    def __init__(self, dimensions):
        self.dim1 = dimensions[0]
        self.dim2 = dimensions[1]
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unflatten(nn.Module):
    def forward(self, x):
        #return x.view
        pass

class Fold(nn.Module):
    def forward(self, x):
        return x.view(-1, 2, int(x.size(1)/2))

#class Print_Size(nn.Module):
    #def forward(self, x):
    #print(x.size())
    #return x


class VAE(nn.Module):
    
    """
        Class for Variational Auto Encoder with Convolutional Layers
    """
    def __init__(self, num_features):
        super(VAE, self).__init__()
        
        #self.Encoder = nn.Sequential(
            #nn.Conv2d(3, num_features, 5),
            #nn.MaxPool2d(kernel_size=2),
            #nn.ReLU(),
            #nn.Conv2d(num_features, num_features*2, 5),
            #nn.MaxPool2d(kernel_size=2),
            #nn.ReLU(),
            #nn.Conv2d(num_features*2, num_features*4, 5),
            #nn.MaxPool2d(kernel_size=2),
            #nn.ReLU(),
            #nn.Conv2d(num_features*4, num_features*8, 5),
            #nn.MaxPool2d(kernel_size=2),
            #nn.ReLU(),
            #Print_Size(),
            #Flatten()
            #)
        
        self.conv1 = nn.Conv2d(3, num_features, 5)
        self.conv2 = nn.Conv2d(num_features, num_features*2, 5)
        self.conv3 = nn.Conv2d(num_features*2, num_features*4, 5)
        self.conv4 = nn.Conv2d(num_features*4, num_features*8, 5)
        
        self.convT1 = nn.ConvTranspose2d(num_features*4, num_fetures*4, 5)
        self.convT2 = nn.ConvTranspose2d(num_features*4, num_fetures*2, 5)
        self.convT3 = nn.ConvTranspose2d(num_features*2, num_fetures, 5)
        self.convT4 = nn.ConvTranspose2d(num_features, 3, 5)
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = Flatten()
        self.fold = Fold()
        self.unflatten = Unflatten()
        

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(3, num_features, 5)

        )
    
    
    def Encoder(self, x):
        x = self.conv1(x)
        x, idx1 = self.pool(x)
        x = self.relu
        x = self.conv2(x)
        x, idx2 = self.pool(x)
        x = self.relu(x)
        x = self.conv3(x)
        x, idx3 = self.pool(x)
        x = self.relu(x)
        x = self.conv4(x)
        x, idx4 = self.pool(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fold(x)
        return x, idx1, idx2, idx3, idx4
    
    def Decoder(self, x):
        
    
        pass


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = self.Encoder(x)
        print(x.size())
        x = x.view(-1, 2, int(x.size(1)/2))
        print(x.size())
        x = self.reparameterize(x[:,0,:], x[:,1,:])
        print(x.size())
        
        return x

























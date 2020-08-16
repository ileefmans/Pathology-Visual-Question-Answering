import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision

import os

# takeout part where tokens=list(tok.word_count)
# use doclist instead because tokenizer already makes tokens obviously


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unflatten(nn.Module):
    def __init__(self, x, num_features=None, dimensions=None):
        super(Unflatten, self).__init__()
        
        self.dimensions = dimensions
        self.num_features = num_features
    
    def forward(self, x):
        if self.num_features==None or self.dimensions==None:
            raise Exception("Mandatory argument not assigned")
        
        return x.view(x.size(0), self.num_features*8, self.dimensions[0], self.dimensions[1])

class Fold(nn.Module):
    def forward(self, x):
        return x.view(-1, 2, int(x.size(1)/2))



class VAE(nn.Module):
    
    """
        Class for Variational Auto Encoder with Convolutional Layers
    """
    def __init__(self, num_features):
        super(VAE, self).__init__()
        
        self.conv1 = nn.Conv2d(3, num_features, 5)
        self.conv2 = nn.Conv2d(num_features, num_features*2, 5)
        self.conv3 = nn.Conv2d(num_features*2, num_features*4, 5)
        self.conv4 = nn.Conv2d(num_features*4, num_features*8, 5)
        
        self.convT1 = nn.ConvTranspose2d(num_features*8, num_features*4, 5)
        self.convT2 = nn.ConvTranspose2d(num_features*4, num_features*2, 5)
        self.convT3 = nn.ConvTranspose2d(num_features*2, num_features, 5)
        self.convT4 = nn.ConvTranspose2d(num_features, 3, 5)
        
        
        self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = Flatten()
        self.fold = Fold()
        self.num_features = num_features
        

    
    
    def Encoder(self, x):
        x = self.conv1(x)
        dim1 = x.size()
        x, idx1 = self.pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        dim2 = x.size()
        x, idx2 = self.pool(x)
        x = self.relu(x)
        x = self.conv3(x)
        dim3 = x.size()
        x, idx3 = self.pool(x)
        x = self.relu(x)
        x = self.conv4(x)
        dim4 = x.size()
        x, idx4 = self.pool(x)
        x = self.relu(x)
        pic_dim = (x.size(2), x.size(3))
        x = self.flatten(x)
        x = self.fold(x)
        idx = [idx1, idx2, idx3, idx4]
        prepool_dim = [dim1, dim2, dim3, dim4]
        return x, idx, pic_dim, prepool_dim
    
    def Decoder(self, x, idx, prepool_dim):
        
        unpool1 = nn.MaxUnpool2d(kernel_size=2)
        unpool2 = nn.MaxUnpool2d(kernel_size=2)
        unpool3 = nn.MaxUnpool2d(kernel_size=2)
        unpool4 = nn.MaxUnpool2d(kernel_size=2)
        
        x = unpool1(x, idx[3], prepool_dim[3])
        x = self.convT1(x)
        x = self.relu(x)
        x = unpool2(x, idx[2], prepool_dim[2])
        x = self.convT2(x)
        x = self.relu(x)
        x = unpool3(x, idx[1], prepool_dim[1])
        x = self.convT3(x)
        x = self.relu(x)
        x = unpool4(x, idx[0], prepool_dim[0])
        x = self.convT4(x)
        x = self.relu(x)
        return x
        


    def reparameterize(self, mu, logvar):
        std= logvar.mul(0.5).exp_()
        eps1 = std.data.new(std.size()).normal_()
        eps = std.data.new(std.size()).normal_()
        return torch.cat((eps1.mul(std).add_(mu), eps.mul(std).add_(mu)), 1)

    def forward(self, x):
        x, idx, dimensions, prepool_dim = self.Encoder(x)
        print("1,", x.size())
        x = self.reparameterize(x[:,0,:], x[:,1,:])
        print("2", x.size())
        unflatten = Unflatten(x, dimensions=dimensions, num_features = self.num_features)
        x = unflatten(x)
        print("3", x.size())
        x = self.Decoder(x, idx, prepool_dim)
        print("4", x.size())
        
        
        return x



#### need to figure out what dimensons to pass to unflatten()





















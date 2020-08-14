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
    def forward(self, x):
        #return x.view
        pass


class VAE(nn.Module):
    
    """
        Class for Variational Auto Encoder with Convolutional Layers
    """
    def __init__(self, num_features):
        super(VAE, self).__init__()
        
        self.Encoder = nn.Sequential(
            nn.Conv2d(3, num_features, 5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features*2, 5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(num_features*2, num_features*4, 5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(num_features*4, num_features*8, 5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            Flatten()
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(3, num_features, 5)

        )
    
    


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
        fc = nn.Linear(x.size(1), x.size(1)*2)
        x = fc(x)
        return x

























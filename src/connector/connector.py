import numpy as np # for transformation
import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer

class Connector(nn.Module):

    def __init__(self):
        super().__init__()

    '''
    def forward(self, x):
        #x = torch.reshape(x, (256, 1, 1, 105))
        slices = [torch.max(x[..., (i*12):((i+1)*12)], keepdim=True, dim=3).values for i in range(8)]
        slices.append(torch.max(x[..., 96:105], keepdim=True, dim=3).values)
        x = torch.cat(slices, dim=3)
        return x
    '''

    def forward(self, x):
        x = torch.reshape(x, (256, 16, 16, 1750))
        return x

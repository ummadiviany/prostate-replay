# imports and installs
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torch.optim import SGD, Adam, ASGD

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import monai
from monai.transforms import (ScaleIntensityRange, Compose, AddChannel, RandSpatialCrop, ToTensor, 
                            RandAxisFlip, Activations, AsDiscrete, Resize, RandRotate, RandFlip, EnsureType,
                             KeepLargestConnectedComponent, CenterSpatialCrop)
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, FocalLoss, GeneralizedDiceLoss, DiceCELoss, DiceFocalLoss
from monai.networks.nets import UNet, VNet, UNETR, SwinUNETR, AttentionUnet
from monai.data import decollate_batch, ImageDataset
from monai.utils import set_determinism
import os
import wandb
from time import time
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from random import sample
from torchvision.transforms import ToPILImage
from torchvision import transforms
import argparse
import pandas as pd

torch.manual_seed(2000)
set_determinism(seed=2000)

wandb_log = False

# ------------------------------------------------------------------------------------------------

# mnist_train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# mnist_test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
# mnist_train_loader = DataLoader(mnist_train_set, batch_size=100, shuffle=True)
# mnist_test_loader = DataLoader(mnist_test_set, batch_size=100, shuffle=True)

f_mnist_train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
f_mnist_test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
f_mnist_train_loader = DataLoader(f_mnist_train_set, batch_size=100, shuffle=True)
f_mnist_test_loader = DataLoader(f_mnist_test_set, batch_size=100, shuffle=True)

# ------------------------------------------------------------------------------------------------

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act='relu', use_bn=False):
        super(LinearLayer, self).__init__()
        self.use_bn = use_bn
        self.lin = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU() if act == 'relu' else act
        if use_bn:
            self.bn = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        if self.use_bn:
            return self.bn(self.act(self.lin(x)))
        return self.act(self.lin(x))

class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)

class BaseModel(nn.Module):
    
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(BaseModel, self).__init__()
        self.f1 = Flatten()
        self.lin1 = LinearLayer(num_inputs, num_hidden, use_bn=True)
        self.lin2 = LinearLayer(num_hidden, num_hidden, use_bn=True)
        self.lin3 = nn.Linear(num_hidden, num_outputs)
        
    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(self.f1(x))))

model = BaseModel(784, 256, 10)

ce_loss = nn.CrossEntropyLoss()

# ------------------------------------------------------------------------------------------------

from elastic_weight_consolidation import ElasticWeightConsolidation
from tqdm import tqdm

ewc = ElasticWeightConsolidation(model, ce_loss, 1e-3)

for _ in range(4):
    for input, target in tqdm(mnist_train_loader):
        ewc.forward_backward_update(input, target)
        
        
ewc.register_ewc_params(mnist_train_set, 100, 300)

for _ in range(4):
    for input, target in tqdm(f_mnist_train_loader):
        ewc.forward_backward_update(input, target)
        
ewc.register_ewc_params(f_mnist_train_set, 100, 300)


def accu(model, dataloader):
    model = model.eval()
    acc = 0
    for input, target in dataloader:
        o = model(input)
        acc += (o.argmax(dim=1).long() == target).float().mean()
    return acc / len(dataloader)


accu(ewc.model, mnist_test_loader)

accu(ewc.model, f_mnist_test_loader)




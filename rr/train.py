# imports and installs
from turtle import right
import torch
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
import argparse
import pandas as pd


torch.manual_seed(2000)
set_determinism(seed=2000)

wandb_log = True

from dataloader import get_dataloader, get_img_label_folds, get_dataloaders
from agg_metrics import print_cl_metrics
# ------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='For training config')

parser.add_argument('--order', type=str, help='order of the dataset domains')
parser.add_argument('--device', type=str, help='Specify the device to use')
parser.add_argument('--optimizer', type=str, help='Specify the optimizer to use')

parser.add_argument('--initial_epochs', type=int, help='No of epochs')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--lr_decay', type=float, help='Learning rate decay factor for each dataset')
parser.add_argument('--epoch_decay', type=float, help='epochs will be decayed after training on each dataset')

parser.add_argument('--replay', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--store_samples', type=int, help='No of samples to store for replay')

# python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:1 --optimizer adam --epochs 100 --lr 1e-3 --lr_decay 1 --epoch_decay 1  --replay --store_samples 5

parsed_args = parser.parse_args()

domain_order = parsed_args.order.split(',')
device = parsed_args.device
optimizer_name = parsed_args.optimizer

initial_epochs = parsed_args.initial_epochs
initial_lr = parsed_args.lr
lr_decay = parsed_args.lr_decay
epoch_decay = parsed_args.epoch_decay
use_replay = parsed_args.replay
store_samples = parsed_args.store_samples

print('-'*100)
print(f"{'-->'.join(domain_order)}")
print(f"Using device : {device}")
print(f"Initially training for {initial_epochs} epochs")
print(f"Using optimizer : {optimizer_name}")

print(f"Inital learning rate : {initial_lr}")
print(f"Using replay : {use_replay}")
print(f"Replay Sample Size : {store_samples}")

print(f"LR decay  : {lr_decay}")
print(f"Epoch decay : {epoch_decay}")
print('-'*100)

# ----------------------------Train Config-----------------------------------------

model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=3,
    ).to(device)


dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
hd_metric = HausdorffDistanceMetric(include_background=False, percentile = 95.)
post_pred = Compose([
    EnsureType(), AsDiscrete(argmax=True, to_onehot=2),
    KeepLargestConnectedComponent(applied_labels=[1], is_onehot=True, connectivity=2)
])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
argmax = AsDiscrete(argmax=True)
dice_ce_loss = DiceCELoss(to_onehot_y=True, softmax=True,)



# ------------------------------Representatoin Replay-------------------------------
import random
model_layer_map = {}
for name, module in model.named_modules():
    model_layer_map[name] = module
    
def get_model_layer(layer_name):
    return model_layer_map[layer_name]

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model_layer_name = 'model.1.submodule.1.submodule.1.submodule.1.submodule.conv.unit2.adn.A'
model_layer = get_model_layer(model_layer_name)

replay_memory = []
l1_loss = nn.L1Loss()

# ------------------------------------WANDB Logging-------------------------------------

# List for datasets will be passed as an argument for this file


config = {
    "Model" : "UNet2D",
    "Seqential Strategy" : f"Latent Represenation Replay with {store_samples} samples for each domain",
    "Replay Strategy" : "Sample same no of channels from the replay memory as of curr batch size",
    "Replay Loss" : "L1 Loss",
    "Domain Ordering" : domain_order,
    # "Batch Training Strategy" : "A batch from current dataset and a batch from episodic memeory are stacked. One backward pass and paramenter update.",
#     "Train Input ROI size" : train_roi_size,
#     "Test Input size" : (1, 320, 320),
#     "Test mode" : f"Sliding window inference roi = {train_roi_size}",
    "Batch size" : "No of slices in original volume",
    "No of volumes per batch" : 1,
    "Initial Epochs" : initial_epochs,
    "Epoch decay" : epoch_decay,
    "Optimizer" : optimizer_name.capitalize(),
    "Scheduler" : "CosineAnnealingLR",
    "Initial LR" : initial_lr,
    "LR decay" : lr_decay,
    "Loss" : "DiceCELoss", 
    "Train Data Augumentations" : "RandSpatialCrop(160,160)",
    "Test Data Preprocess" : "",
    "Train samples" : {"Promise12" : 45, "ISBI" : 63, "Decathlon" : 25, "Prostate158" : 119},
    "Test Samples" : {"Promise12" : 5, "ISBI" : 16, "Decathlon" : 7, "Prostate158" : 20},
    "Pred Post Processing" : "KeepLargestConnectedComponent"
}

if wandb_log:
    wandb.login()
    wandb.init(project="CL_Baselines", entity="vinayu", config = config)
    
batch_size = 1
test_shuffle = True
val_interval = 5
batch_interval = 25
img_log_interval = 15
log_images = False

def train(train_loader : DataLoader, replay : bool = False):
    """
    Inputs : No Inputs
    Outputs : No Outputs
    Function : Trains all datasets and logs metrics to WANDB
    """
    model.train()
    train_start = time()
    epoch_loss = 0
    print('\n')
    
    # Iterating over the dataset
    for i, (imgs, labels) in enumerate(train_loader, 1):

        imgs, labels = imgs.to(device), labels.to(device)
        
        imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
        labels = rearrange(labels, 'b c h w d -> (b d) c h w')

        optimizer.zero_grad()
        preds = model(imgs)

        loss = dice_ce_loss(preds, labels)
        # print(f"DiceCE Loss : {loss.item():.3f}")
        
        if replay:
            # Sample one latent representation from replay memory
            replay_sample = random.sample(replay_memory, 1)[0].to(device)
            # print(f"Replay Sample shape: {replay_sample.shape}")
            
            # Compute L1 loss for representation replay samples and add to total loss
            model_layer.register_forward_hook(get_activation(model_layer_name))
            latent_representation = activation[model_layer_name]
            # print(f"Latent Representation shape : {latent_representation.shape}")
            
            # # Check which one has minimum no of channels
            # min_channels = min(replay_sample.shape[0], latent_representation.shape[0])
            # min_idx = np.argmin([replay_sample.shape[0], latent_representation.shape[0]])
            # max_channels = max(replay_sample.shape[0], latent_representation.shape[0])
            
            # diff = max_channels - min_channels
            # left = diff // 2
            # right = max_channels - left
            
            # if diff % 2 != 0:
            #     right -= 1
            
            # if min_idx == 0:
            #     latent_representation = latent_representation[left:right, :, :, :].to(device)
            # else:
            #     replay_sample = replay_sample[left:right, :, :, :].to(device)
            
            # latent_loss = l1_loss(replay_sample, latent_representation)
            latent_loss = torch.abs(replay_sample.mean() - latent_representation.mean())
            # latent_loss += torch.abs(replay_sample.median() - latent_representation.median())
            # print(f"Latent loss : {latent_loss:.3f}")
            loss += latent_loss
            
        
        preds = [post_pred(i) for i in decollate_batch(preds)]
        preds = torch.stack(preds)
        labels = [post_label(i) for i in decollate_batch(labels)]
        labels = torch.stack(labels)
        
        # Metric scores
        dice_metric(preds, labels)
        hd_metric(preds, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if i % batch_interval == 0:
            print(f"Epoch: [{epoch}/{epochs}], Batch: [{i}/{len(train_loader)}], Loss: {loss.item() :.4f}, \
                  Dice: {dice_metric.aggregate().item() * 100 :.2f}, HD: {hd_metric.aggregate().item() :.2f}")
    
    # Print metrics, log data, reset metrics
    
    print(f"\nEpoch: [{epoch}/{epochs}], Avg Loss: {epoch_loss / len(train_loader) :.3f}, \
              Train Dice: {dice_metric.aggregate().item() * 100 :.2f}, Train HD: {hd_metric.aggregate().item() :.2f}, Time : {int(time() - train_start)} sec")

    log_metrics = {f"{dataset_name.upper()} Train Dice" : dice_metric.aggregate().item() * 100,
                   f"{dataset_name.upper()} Train HD" : hd_metric.aggregate().item(),
                   f"{dataset_name.upper()} Train Loss" : epoch_loss / len(train_loader),
                   # "Learning Rate" : scheduler.get_last_lr()[0],
                   f"Epoch" : epoch }
    if wandb_log:
        wandb.log(log_metrics)
        print(f'Logged training metrics to wandb')


    dice_metric.reset()
    hd_metric.reset()
    scheduler.step()
    
    
def validate(test_loader : DataLoader, dataset_name : str = None):
    """
    Inputs : Testing dataloader
    Outputs : Returns Dice, HD
    Function : Validate on the given dataloader and return the mertics 
    """
    train_start = time()
    model.eval()
    with torch.no_grad():
        # Iterate over all samples in the dataset
        for i, (imgs, labels) in enumerate(test_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
            labels = rearrange(labels, 'b c h w d -> (b d) c h w')

            # preds = model(imgs)
            roi_size = (160, 160)
            preds = sliding_window_inference(inputs=imgs, roi_size=roi_size, sw_batch_size=4,
                                                predictor=model, overlap = 0.5, mode = 'gaussian', device=device)
            
            preds = [post_pred(i) for i in decollate_batch(preds)]
            preds = torch.stack(preds)
            labels = [post_label(i) for i in decollate_batch(labels)]
            labels = torch.stack(labels)

            dice_metric(preds, labels)
            hd_metric(preds, labels)

        val_dice = dice_metric.aggregate().item()
        val_hd = hd_metric.aggregate().item()
        
        dice_metric.reset()
        hd_metric.reset()
        
        print("-"*100)
        print(f"Epoch : [{epoch}/{epochs}], Dataset : {dataset_name.upper()}, Test Avg Dice : {val_dice*100 :.2f}, Test Avg HD : {val_hd :.2f}, Time : {int(time() - train_start)} sec")
        print("-"*100)
        
        if wandb_log and log_images and epoch % img_log_interval == 0:
                preds = torch.stack([argmax(c) for c in preds])
                labels = torch.stack([argmax(c) for c in labels])
                f = make_grid(torch.cat([imgs,labels,preds],dim=3), nrow =2, padding = 20, pad_value = 1)
                images = wandb.Image(ToPILImage()(f.cpu()), caption="Left: Input, Middle : Ground Truth, Right: Prediction")
                wandb.log({f"{metric_prefix}_{dataset_name.upper()} Predictions": images, "Epoch" : epoch})
                print(f'Logged {dataset_name} segmentation predeictions to wandb')
            
        
        return val_dice, val_hd
    
dataloaders_map, dataset_map = get_dataloaders()
    
# -----------------------------------------------------------------------



@torch.no_grad()
def accumulate_replay_memory():
    print('-'*100)
    print("Accumulating replay memory...")
    print(f"Storing {store_samples} latent representations for {dataset_name.capitalize()} to replay buffer")
    
    for _ in range(store_samples):
        imgs, _ = next(iter(dataloaders_map[dataset_name]['train']))
        imgs = imgs.to(device)
        imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
        
        model_layer.register_forward_hook(get_activation(model_layer_name))
        _ = model(imgs)
        latent_representation = activation[model_layer_name]
        print(f"Latent representation shape : {latent_representation.shape}")
        replay_memory.append(latent_representation.cpu())

    print(f"Current replay memory size : {len(replay_memory)}")
    print('-'*100)

optimizer_map ={
    'sgd' : torch.optim.SGD,
    'rmsprop' : torch.optim.RMSprop,
    'adam' : torch.optim.Adam,
}

optimizer_params  = {
    'sgd' : {'momentum' : 0.9, 'weight_decay' : 1e-5, 'nesterov' : True},
    'rmsprop' : {'momentum' : 0.9, 'weight_decay' : 1e-5},
    'adam' : {'weight_decay': 1e-5,},   
}

test_metrics = []
# epochs_list = [100, 64, 40, 26]

for i, dataset_name in enumerate(domain_order, 1):
    
    epochs = int(initial_epochs * (epoch_decay**(i-1))) 
    # epochs = epochs_list[i-1]
    lr = initial_lr * (lr_decay**(i-1))
    optimizer = optimizer_map[optimizer_name](model.parameters(), lr =lr, **optimizer_params[optimizer_name])    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, verbose=True)

    train_loader = dataloaders_map[dataset_name]['train']
    test_dataset_names = ['prostate158', 'isbi', 'promise12', 'decathlon']

    metric_prefix  = i
    for epoch in range(1, epochs + 1):   
        
        if i == 1:
            train(train_loader = train_loader, replay=False)
        else:
            if use_replay:
                train(train_loader = train_loader, replay=True)

        if epoch % val_interval == 0:
            test_metric = []
            for dname in test_dataset_names:
                val_dice, val_hd = validate(test_loader = dataloaders_map[dname]['test'], dataset_name = dname)
                
                log_metrics = {}
                log_metrics[f'Epoch'] = epoch
                log_metrics[f'{metric_prefix}_{dname}_curr_dice'] = val_dice*100 
                log_metrics[f'{metric_prefix}_{dname}_curr_hd'] = val_hd

                if wandb_log:
                    # Quantiative metrics
                    wandb.log(log_metrics)
                    print(f'Logged {dname} test metrics to wandb')
                    
                test_metric.append(val_dice*100)
                    
    test_metrics.append(test_metric)
    
    # Add 5 latent representations to replay memory
    accumulate_replay_memory()    


cl_metrics = print_cl_metrics(domain_order, test_dataset_names, test_metrics)
if wandb_log:
    wandb.log(cl_metrics)
    print(f'Logged CL metrics to wandb')

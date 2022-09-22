# imports and installs
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

torch.manual_seed(2000)
set_determinism(seed=2000)

def get_img_label_folds(img_paths, label_paths):
    
    fold = list(range(0,len(img_paths)))
    fold = sample(fold, k=len(fold))
    fold_imgs = [img_paths[i] for i in fold]
    fold_labels = [label_paths[i] for i in fold]
    return fold_imgs, fold_labels

def get_dataloader(img_paths : list, label_paths : list, train : bool):
    
    if train:
        ttset = "train"
    else:
        ttset = "test"
        
    dataset = ImageDataset(img_paths, label_paths,
                            transform=Compose(transforms_map[f'{ttset}_img_transform']), 
                            seg_transform=Compose(transforms_map[f'{ttset}_label_transform']))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return  dataloader

batch_size = 1
train_roi_size = 160
resize_dim = 256

# Transforms for images & labels
transforms_map = {
        "train_img_transform" : [
            AddChannel(),
            # Resize(spatial_size=(resize_dim, resize_dim, -1)),
            # CenterSpatialCrop([train_roi_size, train_roi_size, -1]),
            RandSpatialCrop(roi_size= train_roi_size, random_center = True, random_size=False),
            ToTensor()
            ],
        "train_label_transform" : [
            AddChannel(),
            # Resize(spatial_size=(resize_dim, resize_dim, -1)),
            # CenterSpatialCrop([train_roi_size, train_roi_size, -1]),
            RandSpatialCrop(roi_size= train_roi_size, random_center = True, random_size=False),
            AsDiscrete(threshold=0.5),
            ToTensor()
            ],
        "test_img_transform" : [
            AddChannel(),
            # Resize(spatial_size=(resize_dim, resize_dim, -1)),
            # CenterSpatialCrop([train_roi_size, train_roi_size, -1]),
            ToTensor()
            ],
        "test_label_transform" : [
            AddChannel(),
            # Resize(spatial_size=(resize_dim, resize_dim, -1)),
            # CenterSpatialCrop([train_roi_size, train_roi_size, -1]),
            AsDiscrete(threshold=0.5),
            ToTensor()
            ],
    }


# 1. Image & Label paths
dataset_map = {
        "promise12" : {
            "data_dir" : "../datasets/promise12prostatealigned/",
            "test_size" : 0.1,
            'test' :  {'images' : [], 'labels' : []},
            'train' :  {'images' : [], 'labels' : []}
            
            },
        "decathlon" : {
            "data_dir" : "../datasets/decathlonprostatealigned/",
            "test_size" : 0.2,
            'test' :  {'images' : [], 'labels' : []},
            'train' :  {'images' : [], 'labels' : []}
            },
        "isbi" : {
            "data_dir" : "../datasets/isbiprostatealigned/",
            "test_size" : 0.2,
            'test' :  {'images' : [], 'labels' : []},
            'train' :  {'images' : [], 'labels' : []}
            },
        "prostate158" : {
                "data_dir" : "../datasets/prostate158aligned/",
                "test_size" : 0.15,
                'test' :  {'images' : [], 'labels' : []},
                'train' :  {'images' : [], 'labels' : []}
                }
    }

def get_dataloaders():
    
    for dataset in dataset_map:
        # print(f"------------{dataset}------------")
        data_dir = dataset_map[dataset]['data_dir']

        img_paths = glob(data_dir + "imagesTr/*.nii")
        label_paths = glob(data_dir + "labelsTr/*.nii")
        img_paths.sort()
        label_paths.sort()
        
        # 2. Folds

        images_fold, labels_fold  = get_img_label_folds(img_paths, label_paths)
        
        # print("Number of images: {}".format(len(images_fold)))
        # print("Number of labels: {}".format(len(labels_fold)))
        
        # Get train and test sets
        # 3. Split into train - test
        train_idx = int(len(images_fold) * (1 - dataset_map[dataset]['test_size']))
        
        # Store train & test sets 
        
        dataset_map[dataset]['train']['images'] = images_fold[:train_idx]
        dataset_map[dataset]['train']['labels'] = labels_fold[:train_idx]
        
        dataset_map[dataset]['test']['images'] = images_fold[train_idx:]
        dataset_map[dataset]['test']['labels'] = labels_fold[train_idx:]
        
        dataloaders_map = {}

    for dataset in dataset_map:
        # print(f"------------{dataset}------------")
        dataloaders_map[dataset] = {}
        for ttset in ['train', 'test']:
            
            if ttset == 'train':
                train = True
            else:
                train = False
            
            dataloaders_map[dataset][ttset] = get_dataloader(img_paths = dataset_map[dataset][ttset]['images'],
                                                            label_paths = dataset_map[dataset][ttset]['labels'],
                                                            train = train)
            
            # print(f"""No of samples in {dataset}-{ttset} : {len(dataloaders_map[dataset][ttset])}""")

    # 7. That's it
    
    return dataloaders_map, dataset_map


if __name__ == "__main__":
    start = time()
    
    dataloaders_map, dataset_map = get_dataloaders()
    print("Done")
    # print(f"Data loaders map: {dataloaders_map}")
    # print(f"Dataset map: {dataset_map}")
    
    priomise12_train = dataloaders_map['promise12']['train']
    imgs, labels = next(iter(priomise12_train))
    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
    labels = rearrange(labels, 'b c h w d -> (b d) c h w')
    print(f"Image shape : {imgs.shape}")
    print(f"Label shape : {labels.shape}")

    img_no = 8
    plt.figure(figsize=(6*3,6*1))
    plt.subplot(1,3,1)
    plt.imshow(imgs[img_no,0], cmap='gray')
    plt.axis('off')
    plt.title('Image')
    plt.subplot(1,3,2)
    plt.imshow(labels[img_no,0], cmap='gray')
    plt.axis('off')
    plt.title('Label')
    plt.subplot(1,3,3)
    plt.imshow(imgs[img_no,0], cmap='gray')
    plt.imshow(labels[img_no,0], 'copper', alpha=0.2)
    plt.axis('off')
    plt.title('Overlay')
    # plt.show()
    plt.savefig('promise12.png')
    
    print(f"Completed in: {time() - start:.1f} seconds")
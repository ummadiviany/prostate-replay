# imports and installs
import json
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
from monai.data import decollate_batch
from monai.utils import set_determinism
import os
import wandb
from time import time
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from random import sample
from torchvision.transforms import ToPILImage

from dataset import ImageDataset

torch.manual_seed(2000)
set_determinism(seed=2000)

def get_img_label_folds(img_paths, label_paths):
    
    fold = list(range(0,len(img_paths)))
    fold = sample(fold, k=len(fold))
    fold_imgs = [img_paths[i] for i in fold]
    fold_labels = [label_paths[i] for i in fold]
    return fold_imgs, fold_labels

def get_dataloader(img_paths : list, label_paths : list, train : bool = True):
    
    if train:
        ttset = "train"
    else:
        ttset = "test"
        
    dataset = ImageDataset(img_paths, label_paths,
                            transform=Compose(transforms_map[f'{ttset}_img_transform']), 
                            seg_transform=Compose(transforms_map[f'{ttset}_label_transform']))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    
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
        
        # print(f"------------{dataset}------------")
        # print(f"First train image: {dataset_map[dataset]['train']['images'][0]}")
        # print(f"Last train image: {dataset_map[dataset]['train']['images'][-1]}")
        
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
    # print(f"Data loaders map: {dataloaders_map}")
    # print(f"Dataset map: {dataset_map}")
    # label_class_map = {}
    # for dataset_name in dataset_map:
    #     print(f"------------{dataset_name}------------")
    #     label_class_map[dataset_name] = {}
        
    #     train_loader = get_dataloader(img_paths = dataset_map[dataset_name]['train']['images'],
    #                                   label_paths= dataset_map[dataset_name]['train']['labels'],
    #                                   train=False)
        
    #     for i, (img, label) in enumerate(train_loader):
    #         img_path = dataset_map[dataset_name]['train']['images'][i]
    #         print(f"Image 0 : {img_path}")
    #         print(f"Image 0 shape: {img.shape}")
    #         ones_percent = (label.sum() / label.numel()) * 100
    #         print(f"Ones percent: {ones_percent:.2f}")
    #         label_class_map[dataset_name][i] = ones_percent.item()
    #         # break
        
    #     # Sort the dict by value 
    #     label_class_map[dataset_name] = dict(sorted(label_class_map[dataset_name].items(), key=lambda item: item[1], reverse=True))
    
    # import json
    # with open('label_class_map.json', 'w') as fp:
    #     json.dump(label_class_map, fp)
    
    # label_class_map = json.load(open('label_class_map.json'))
    # for dataset_name in label_class_map:
    #     print(f"------------{dataset_name}------------")
    #     idxs = list(label_class_map[dataset_name].keys())
    #     idxs = list(map(int, idxs[:5]))
    #     print(f"Top 5 indices: {idxs}")
        # for i, key in enumerate(label_class_map[dataset_name]):
        #     print(f"Image path {key} : {dataset_map[dataset_name]['train']['images'][int(key)]}")
        #     print(f"Ones percent: {label_class_map[dataset_name][key]:.2f}")
            
        #     if i == 5:
        #         break
        
    
    # index_to_filenames = {}
    # for dataset_name in dataset_map:
    #     index_to_filenames[dataset_name] = {}
    #     imgs = dataset_map[dataset_name]['train']['images']
    #     for i, img in enumerate(imgs):
    #         index_to_filenames[dataset_name][i] = img

    # import json
    # with open('index_to_filenames.json', 'w') as fp:
    #     json.dump(index_to_filenames, fp)
        
    print(f"Completed in: {time() - start:.1f} seconds")
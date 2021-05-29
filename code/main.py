import os, sys
import yaml
import random
import argparse
import numpy as np
from tqdm import tqdm
from pprint import pprint

import albumentations
import albumentations.pytorch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision import utils, datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch_optimizer as optim

import timm
import madgrad

import wandb
from torchsummary import summary as summary_
from torch.utils.tensorboard import SummaryWriter

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from adamp import AdamP

from model import *
from loss import *
from dataset import MaskDataset, AugMix
from train import train_with_val, multi_label_train_with_val, multi_label_train

# argparser
parser = argparse.ArgumentParser(description='Create labeled .csv file')
parser.add_argument('--config', required=False, default='Base', help='Enter the config name to apply.')

args = parser.parse_args()
config_name = args.config

# Set config
with open("config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)[config_name]

model_name = cfg['model_name']
num_epochs = cfg['num_epochs']
multi_head = cfg['multi_head']
num_workers = cfg['num_workers']
learning_rate = cfg['learning_rate']
weight_decay = cfg['weight_decay']
optimizer_name = cfg['optimizer']
scheduler_name = cfg['scheduler']
train_log_interval = cfg['train_log_interval']
cross_valid = cfg['cross_valid']
kfold = cfg['kfold']
whole_label_path = cfg['whole_label_path']
train_label_path = cfg['train_label_path']
val_label_path = cfg['val_label_path']
batch_size = cfg['batch_size']
seed = cfg['seed']
criterion_name = cfg['criterion']
if criterion_name == 'crossentropy':
    criterion = nn.CrossEntropyLoss()
elif criterion_name == 'focal':
    criterion = FocalLoss()
elif criterion_name == 'labelsmoothing':
    criterion = LabelSmoothingLoss()
else:
    criterion = F1Loss()
    

# For Mixed Precision
scaler = torch.cuda.amp.GradScaler(enabled=True)
        
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Set seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

train_transform = albumentations.Compose([
    albumentations.OneOf([
        albumentations.HorizontalFlip()
    ]), 
    albumentations.Resize(384, 512),
    albumentations.Normalize(mean=(0.560, 0.524, 0.501), std=(0.233, 0.243, 0.245)),
    albumentations.pytorch.transforms.ToTensorV2()])

val_transform = albumentations.Compose([
    albumentations.Resize(384, 512),
    albumentations.Normalize(mean=(0.560, 0.524, 0.501), std=(0.233, 0.243, 0.245)),
    albumentations.pytorch.transforms.ToTensorV2()]) 

# Get Dataloader
def get_dataloader(path, transform, shuffle):
    dataset = MaskDataset(label_path=path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=shuffle
    )
    
    return len(dataset), loader

# Get Optimizer and Scheduler
def get_optimizer(model, optimizer_name, scheduler_name):
    if optimizer_name == 'Adam':
            optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamP':
        optimizer = AdamP(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'MADGRAD':
        optimizer = madgrad.MADGRAD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Ranger(model.parameters(), lr=learning_rate, alpha=0.6, k=10)
        
    if scheduler_name == 'step':
        scheduler = StepLR(optimizer, 10, gamma=0.5)
    elif scheduler_name == 'reduce':
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=0.)
    
    return optimizer, scheduler

# Set wandb
wandb.init(config={"batch_size": batch_size,
                   "lr"        : learning_rate,
                   "epochs"    : num_epochs,
                   "backborn"  : cfg["model_name"],
                   "criterion_name" : "CrossEntropyLoss"})

# summary_(model, (3, 384, 512), batch_size=16)

# Train
if cross_valid:
    for k in range(kfold):
        print(f'\n[{k + 1}-Fold] training start------------------------------------\n')
        
        model = MultiFCModel(model_name, True)
        
        optimizer, scheduler = get_optimizer(model, optimizer_name, scheduler_name)

        train_label_path = train_label_path + f'{k}_fold_train_label.csv'
        _, train_loader = get_dataloader(train_label_path, train_transform, True)

        val_label_path = val_label_path + f'{k}_fold_val_label.csv'
        val_set_length, val_loader = get_dataloader(val_label_path, val_transform, False)

        if multi_head:
            multi_label_train_with_val(
                model_name, 
                k, 
                model, 
                criterion, 
                optimizer, 
                train_loader, 
                val_loader, 
                scheduler, 
                num_epochs, 
                train_log_interval, 
                val_set_length, 
                scaler, 
                batch_size
            )
        else:
            train_with_val(
                model_name, 
                k, 
                model, 
                criterion, 
                optimizer, 
                train_loader, 
                val_loader, 
                scheduler, 
                num_epochs, 
                train_log_interval, 
                val_set_length, 
                scaler, 
                batch_size
            )

else:
    _, train_loader = get_dataloader(whole_label_path, train_transform, True)
    model = MultiFCModel(model_name, True)
    
    optimizer, scheduler = get_optimizer(model, optimizer_name, scheduler_name)
    
    multi_label_train(
        model_name, 
        model, 
        criterion, 
        optimizer, 
        train_loader, 
        scheduler, 
        num_epochs, 
        train_log_interval, 
        scaler, 
        batch_size
    )
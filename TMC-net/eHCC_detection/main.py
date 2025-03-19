import random
import shutil
import os
import glob

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, random_split, DataLoader
import torchvision
from torchvision import datasets, models, transforms

from utils import train, valid, save_model, metric
import config
from ResNet import Net

torch.set_num_threads(64)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = dataset.targets
        self.classes = dataset.classes
        self.transform = dataset.transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]      
        return x, y 

    def __len__(self):
        return len(self.indices)

class_names = ['DN', 'HCC']

# model config
model_config = config.model_config
patch_size = model_config['patch_size']
batch_size = model_config['batch_size']
gpu = model_config['gpu']
lr = model_config['lr']
epochs = model_config['epochs']

if gpu >= 0 and torch.cuda.is_available():
    device = torch.device(gpu)
else:
    device = torch.device('cpu')

setup_seed(42)
data_transforms = {
    'train': transforms.Compose([transforms.Resize(patch_size*2), transforms.RandomVerticalFlip(), 
                                 transforms.RandomHorizontalFlip(), transforms.RandomRotation(180), transforms.ToTensor(),
                                 transforms.Normalize([0.7936, 0.4834, 0.6893],[0.1109, 0.1435, 0.0989])]),
    'test': transforms.Compose([transforms.Resize(patch_size*2), transforms.ToTensor(),
                                transforms.Normalize([0.7936, 0.4834, 0.6893],[0.1109, 0.1435, 0.0989])])}

# load dataset and split
train_path = '../data_stage1/Train'
full_dataset = datasets.ImageFolder(root=train_path, transform=data_transforms['train'])
train_size = int(0.8*len(full_dataset))
val_size = len(full_dataset)-train_size
train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])
train_dataset = CustomSubset(full_dataset, train_indices)
val_dataset = CustomSubset(full_dataset, val_indices)
val_dataset.transform = data_transforms['test']

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=32)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=32)

model_raw_large = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model_raw_small = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model = Net(model_raw_large, model_raw_small)
model.to(device)
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=False)

start_epoch = 0
best_acc = 0
    
log_dir = './final_model'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_dir_txt = './final_model/res.txt'

for epoch in range(start_epoch+1, epochs+1):
    train_loss, train_acc, train_f1, train_roc, train_prc = train(model, device, train_loader, optimizer, criterion, epoch)
    val_loss, val_acc, val_f1, val_roc, val_prc = valid(model, device, val_loader, criterion, epoch)
    best_acc = save_model(val_acc, best_acc, log_dir, epoch, model, optimizer)
    res = 'Epoch: {:05d}, Train loss:{:.3f}, Valid loss: {:.3f}, \n'.format(epoch, train_loss, val_loss)
    res = res + 'Train\nROC-AUC: {:.3f}, PR-AUC: {:.3f}, '.format(train_roc, train_prc)
    res = res + 'Accuracy: {:.3f}, F1: {:.3f} \n'.format(train_acc, train_f1)
    res = res + 'Val\nROC-AUC: {:.3f}, PR-AUC: {:.3f}, '.format(val_roc, val_prc)
    res = res + 'Accuracy: {:.3f}, F1: {:.3f} \n'.format(val_acc, val_f1)
    print(res)
    with open(log_dir_txt, "a") as f:
        f.write(res+'\n')
        f.close()


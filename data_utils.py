import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.Resize(tuple(config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['data']['normalize_mean'], std=config['data']['normalize_std'])
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root=config['paths']['data_root'], train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=config['paths']['data_root'], train=False, download=True, transform=transform
    )
    
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    split = int(config['data']['val_split'] * len(train_dataset))
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_dataset = Subset(train_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, val_loader, test_loader
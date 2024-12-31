from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
"""
Updated by Tianyi Wang
"""

# Add function for '.npy' crop image
class NpyDataset(Dataset):

    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.file_paths = [os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith('.npy')]

    def  __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = np.load(file_path)
        image = torch.tensor(image,dtype=torch.float32) # convert to tensor
        if self.transform:
            image = self.transform(image)
        return image

# assume totally random split by tiles for now (Temporary!!! need to be updated for later patient level for TCGA)
def split_dataset(dataset, split_ratio= 0.8):
    train_size = int(split_ratio*len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])

def load_data(root_path, dir, batch_size, phase):

    transform_dict = {
        'src': transforms.Compose([
            #transforms.Resize((256, 256)), # pre-scaled image, removed
            #transforms.RandomCrop((224, 224)),# pre-scaled image, removed
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'tar': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'test': transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])}

    dataset = NpyDataset(root_path + dir, transform = transform_dict[phase])

    # test for splitting for now (Required updated!!!!!)
    if phase == 'src':
        train_dataset, val_dataset = split_dataset(dataset, split_ratio=0.8)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
        return train_loader, val_loader

    elif phase == 'tar':
        # Split target dataset into training (80%) and ignore the rest for now
        train_dataset, _ = split_dataset(dataset, split_ratio=0.8)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
        return train_loader  # Only return the training loader
    else:
        # Return test DataLoader without splitting
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
        return data_loader

def load_train(root_path, dir, batch_size, phase):
    transform_dict = {
        'src': transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ]),
        'val': transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])}
    dataset = NpyDataset(root_path + dir, transform = transform_dict[phase])

    # Handel validation phase separately
    if phase == 'val':
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
        return val_loader
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=(phase == 'src'), drop_last=False,
                                  num_workers=4)
        return train_loader

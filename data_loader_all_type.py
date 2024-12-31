from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Dataset
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

def load_data(root_path, dir, batch_size, phase):

    transform_dict = {
        'src': transforms.Compose([
            #transforms.Resize((256, 256)), # pre-scaled image, removed
            #transforms.RandomCrop((224, 224)),# pre-scaled image, removed
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'tar': transforms.Compose([
            #transforms.Resize((256, 256)),
            #transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'test': transforms.Compose([
            #transforms.Resize((256, 256)),
            #transforms.CenterCrop((224, 224)),
            #transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])}

    dataset = NpyDataset(root_path + dir, transform = transform_dict[phase])
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = (phase !='test'), drop_last = False, num_workers = 4)
    return data_loader

def load_train(root_path, dir, batch_size, phase):
    transform_dict = {
        'src': transforms.Compose(
            [#transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             #transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ]),
        'val': transforms.Compose(
            [#transforms.Resize(224),
             #transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])}
    dataset = NpyDataset(root_path + dir, transform = transform_dict[phase])
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = (phase == 'src'), drop_last=False, num_workers=4)
    return train_loader

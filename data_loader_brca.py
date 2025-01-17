
import random
import os
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset

from image_check import file_path

"""
Updated by Tianyi Wang
"""

set_seed = 1007
torch.manual_seed(set_seed)
np.random.seed(set_seed)
random.seed(set_seed)

# Add function for '.npy' crop image
class NpyDataset(Dataset):

    def __init__(self, root_path, transform=None, labeled = True, is_target=False, TCGA_unique_match = None):
        self.root_path = root_path
        self.transform = transform
        self.labeled = labeled
        self.is_target = is_target

        # Add subfolder to file mapping
        self.file_paths = []
        self.labels = [] if labeled else None

        if is_target: # target domain: grouped by patient ID, no labels
            self.patient_to_files = {}
            self.unique_patient_ids = []
            if TCGA_unique_match:
                print(f"Reading all patient ID from {TCGA_unique_match}")
                df=pd.read_csv(TCGA_unique_match)
                self.unique_patient_ids = df['Case.ID'].unique().tolist()
                print(f"Extracted {len(self.unique_patient_ids)} unique patient IDs")

            subfolders = [
                os.path.join(root_path,subfolder)
                for subfolder in os.listdir(root_path)
                if os.path.isdir(os.path.join(root_path,subfolder))
            ]
            for subfolder_path in subfolders:
                files = [
                    os.path.join(subfolder_path, f)
                    for f in os.listdir(subfolder_path)
                    if f.endswith('.npy')
                ]

                for file_path in files:
                    first_file = os.path.basename(file_path)
                    patient_id = '-'.join(first_file.split('-')[:3]) # TCGA-XX-XXXX

                    # only keep those patient that has a valid RNA-seq match
                    if patient_id in self.unique_patient_ids:
                        if patient_id not in self.patient_to_files :
                            self.patient_to_files[patient_id] = []
                        self.patient_to_files[patient_id].append(file_path)
                    else:
                        print(f"Invalid patient ID not in matching file:{patient_id}")

            self.file_paths=[
                file for files in self.patient_to_files.values() for file in files
            ]

            print(f"Total Unique patient ID files loaded: {len(self.file_paths)}")

        else: #Source Domain: Use subfolder names as label
            self.subfolder_paths = [
                os.path.join(root_path,subfolder)
                for subfolder in os.listdir(root_path)
                if os.path.isdir(os.path.join(root_path,subfolder))
            ]
            self.label_mapping = (
                {subfolder: idx for idx, subfolder in enumerate(os.listdir(root_path))}
                if labeled
                else None
            )

            for subfolder_path in self.subfolder_paths:
                files = [
                    os.path.join(subfolder_path, f)
                    for f in os.listdir(subfolder_path)
                    if f.endswith('.npy')
                ]
                self.file_paths.extend(files)
                if labeled:
                    subfolder_name = os.path.basename(subfolder_path)
                    label = self.label_mapping[subfolder_name]
                    self.labels.extend([label] * len(files))

    def  __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = np.load(file_path)

        image = torch.tensor(image,dtype=torch.float32) # convert to tensor

        # Updated: Crop file format:H, W, C; required format: C, H, W
        #(double check this)
        if len(image.shape) == 3 and image.shape[-1] == 3: #If single-channel(H,W)
            image = image.permute(2, 0 , 1) # convert to C, H, W

        if self.transform:
            image = self.transform(image)

        if self.labeled and not self.is_target:
            label = self.labels[idx]
            label = torch.tensor(label,dtype = torch.long)
            return image, label
        else:
            return image

#Add functions that split dataset at patient level (TCGA)
## make change to check 8 train limit and 2 val limit
def split_by_patient(dataset,split_ratio=0.8, train_test_num = 8, val_test_num =2):

    patient_ids = list(dataset.patient_to_files.keys())

    # Add fixed seed
    np.random.seed(set_seed)
    np.random.shuffle(patient_ids)

    train_size = int(split_ratio * len(patient_ids))
    train_patient_ids = patient_ids[:train_size]
    val_patient_ids = patient_ids[train_size:]

    # updated  for test (8, 2, 25-01-16)
    train_patient_ids = patient_ids[:train_test_num]
    val_patient_ids = patient_ids[:val_test_num]

    train_files = [file for pid in train_patient_ids for file in dataset.patient_to_files[pid]]
    val_files = [file for pid in val_patient_ids for file in dataset.patient_to_files[pid]]


    #Map file paths back to indices for subset
    train_indices = [dataset.file_paths.index(f) for f in train_files]
    val_indices = [dataset.file_paths.index(f) for f in val_files]



    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Test if it is 10 patient 01-16
    print(f"Selected {len(train_patient_ids)} patients for training.")
    print(f"Selected {len(val_patient_ids)} patients for validation.")

    return train_dataset, val_dataset



def load_data(root_path, dir, batch_size, phase, is_target = False, TCGA_unique_match = None):

    transform_dict = {
        'src': transforms.Compose([
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

    dataset = NpyDataset(root_path + dir,
                         transform = transform_dict[phase],
                         labeled = (phase == 'src'),
                         is_target = is_target,
                         TCGA_unique_match  = TCGA_unique_match
                         )

    # test for splitting for now (Required updated!!!!!)
    if phase == 'src':
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        generator = torch.Generator().manual_seed(set_seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size],generator=generator)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )
        return train_loader, val_loader

    elif phase == 'tar':

        train_dataset, val_dataset = split_by_patient(dataset, split_ratio=0.8,  train_test_num = 8, val_test_num =2)

        train_loader = DataLoader(
            dataset = train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4
        )
        val_loader = DataLoader(
            dataset =val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )
        return train_loader, val_loader

    else:

        # Return test DataLoader without splitting
        data_loader = DataLoader(
            dataset =dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )
        return data_loader


import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class NpyDataset(Dataset):
    """
    Custom Dataset for loading .npy files.
    Each .npy file is expected to contain a single data sample.
    """
    def __init__(self, root_path, transform=None):
        self.file_paths = [os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load .npy file
        sample = np.load(self.file_paths[idx])
        # Convert to torch tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        if self.transform:
            sample = self.transform(sample)
        return sample

def load_data_npy(root_path, dir, batch_size, phase):
    """
    Load data for .npy files and return a DataLoader.
    """
    transform_dict = {
        'src': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Example normalization
        ]),
        'tar': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Example normalization
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Example normalization
        ])
    }

    dataset = NpyDataset(root_path=os.path.join(root_path, dir), transform=transform_dict[phase])
    shuffle = phase != 'test'  # Shuffle for training and validation, not for testing
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=4)
    return data_loader

# Usage Example
if __name__ == "__main__":
    root_path = "path_to_data_root"
    target_dir = "target_domain"
    phase = "tar"  # "src" for source domain, "tar" for target domain, "test" for testing
    batch_size = 32

    target_loader = load_data_npy(root_path, target_dir, batch_size, phase)
    for batch in target_loader:
        print(batch.shape)  # Check batch shape

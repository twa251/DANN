import os
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset

class NpyDataset(Dataset):
    def __init__(self, root_path, transform=None):
        """
        Custom Dataset for loading .npy files.
        Args:
            root_path (str): Directory containing .npy files.
            transform (callable, optional): Optional transform to apply to each sample.
        """
        self.file_paths = []
        for subdir, _, files in os.walk(root_path):  # Traverse all subdirectories
            for file in files:
                if file.endswith('.npy'):
                    self.file_paths.append(os.path.join(subdir, file))
        
        if not self.file_paths:
            raise ValueError(f"No .npy files found in {root_path}")
        
        self.transform = transform

        self._validate_file_shapes() #check dim for all files
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the .npy file
        sample = np.load(self.file_paths[idx])
        if len(sample.shape) == 3 and sample.shape[-1] == 3:  # [H, W, C] -> [C, H, W]
            sample = np.transpose(sample, (2, 0, 1))
        elif len(sample.shape) != 3 or sample.shape[0] != 3:  # Validate 3-channel data
            raise ValueError(f"Invalid .npy file shape: {sample.shape}. Expected [C, H, W] with 3 channels.")
        
        sample = torch.tensor(sample, dtype=torch.float32)  # Convert to PyTorch tensor
        if self.transform:
            sample = self.transform(sample)
        return sample, 0  # Return a dummy label (0) for compatibility
    def _validate_file_shapes(self):
        """
        Validate dimensions of all .npy files in the dataset.
        """
        for file_path in self.file_paths:
            data = np.load(file_path)
            if len(data.shape) == 3 and data.shape[-1] == 3:  # [H, W, C]
                continue
            elif len(data.shape) == 3 and data.shape[0] == 3:  # [C, H, W]
                continue
            else:
                raise ValueError(f"Inconsistent shape in file {file_path}: {data.shape}. Expected 3D tensor with 3 channels.")
        
def load_data(root_path, dir, batch_size, phase, use_npy=False):
    """
    Load data for training, validation, or testing.
    Args:
        root_path (str): Root directory path.
        dir (str): Subdirectory containing the data.
        batch_size (int): Batch size.
        phase (str): Phase ('src', 'tar', 'test').
        use_npy (bool): Whether to load .npy files instead of image folders.
    Returns:
        DataLoader: PyTorch DataLoader for the specified dataset.
    """
    if use_npy:
        # Load .npy files
        transform = transforms.Compose([
            transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-6))  # Normalize for numerical stability
        ])
        dataset = NpyDataset(root_path=os.path.join(root_path, dir), transform=transform)
    else:
        # Load image data using ImageFolder
        transform_dict = {
            'src': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'tar': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        dataset = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform_dict[phase])
    
    # Create DataLoader
    shuffle = phase != 'test'  # Shuffle for training and validation, not for testing
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=4)
    return data_loader

# Libraries
# Augmentation Class
from capture24.augmentation import Augment
from torch.utils.data import Dataset
import torch
import numpy as np


class C24_Dataset(Dataset):
    def __init__(self, X, Y, idx_to_label, label_to_idx, exclude_classes=None, augs=None):
        """
        Dataset for Capture-24 accelerometer data.
        
        Args:
            X: numpy array of shape [N, T, C] (samples, time, channels)
            Y: numpy array of shape [N] (labels)
            idx_to_label: dict mapping index to label name
            label_to_idx: dict mapping label name to index
            exclude_classes: list of class indices or names to exclude (None = exclude none)
        """
        self.idx_to_label = idx_to_label
        self.label_to_idx = label_to_idx
        self.original_size = len(Y)
        
        # Apply filtering if exclude_classes is provided
        if exclude_classes is not None:
            # Convert class names to indices if needed
            exclude_idx = []
            for c in exclude_classes:
                if isinstance(c, str):
                    exclude_idx.append(int(label_to_idx[c]))
                else:
                    exclude_idx.append(int(c))
            
            mask = ~np.isin(Y, exclude_idx)
            X = X[mask]
            Y = Y[mask]
        
        # Convert to torch tensors
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()
        self.exclude_classes = exclude_classes
        self.filtered_size = len(self.Y)
        self.augs = Augment()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # X will return a tensor of shape [T, C] and Y will return a tensor of shape [1]
        if self.augs:
            return self.augs(self.X[index]), self.Y[index]
        return self.X[index], self.Y[index]
    
    def get_class_distribution(self):
        """Return a dict with count per class."""
        unique, counts = torch.unique(self.Y, return_counts=True)
        return {self.idx_to_label[str(int(u))]: int(c) for u, c in zip(unique, counts)}


import sys, os
import numpy as np
import math
import random
import itertools
import copy

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch

class CustomDataset(Dataset):
    """Custom Dataset.

    Attributes:
        x: A PyTorch tensor for x features of data.
        y: A PyTorch tensor for y features (true labels) of data.
        z: A PyTorch tensor for z features (sensitive attributes) of data.
    """
    def __init__(self, x_tensor, y_tensor, z_tensor):
        """Initializes the dataset with torch tensors."""
        
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor
        
    def __getitem__(self, index):
        """Returns the selected data based on the index information."""
        
        return (self.x[index], self.y[index], self.z[index])

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.x)
    
    

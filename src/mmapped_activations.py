import torch
import json
import os
import numpy as np
from torch.utils.data import Dataset

class MemoryMappedActivationsDataset(Dataset):
    def __init__(self, data_dir, layer_name, max_size=None):
        self.data_dir = data_dir
        self.layer_name = layer_name
        self.metadata_file = os.path.join(data_dir, f"{layer_name}_metadata.json")
        self.tensor_file = os.path.join(data_dir, f"{layer_name}_tensors.npy")
        
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.mmap = np.load(self.tensor_file, mmap_mode='r')
        if max_size is not None:
            self.metadata['filenames'] = self.metadata['filenames'][:max_size]
            self.metadata['tensor_shapes'] = self.metadata['tensor_shapes'][:max_size]
            self.mmap = self.mmap[:max_size]
        self.activation_shape = self._get_activation_shape()

    def _get_activation_shape(self):
        return self.metadata['tensor_shapes'][0]

    def __len__(self):
        return len(self.metadata['filenames'])
    
    def __getitem__(self, idx):
        filename = self.metadata['filenames'][idx]
        tensor_shape = self.metadata['tensor_shapes'][idx]
        
        # Get the flattened tensor data
        tensor_data = self.mmap[idx]
        
        # Reshape the tensor data to its original shape
        tensor = torch.from_numpy(tensor_data.reshape(tensor_shape))
        
        return filename, tensor
 
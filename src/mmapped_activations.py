from typing import Optional
import pathlib
import argparse
import torch
from hooked_model import WhisperActivationCache
from autoencoder import AutoEncoder, init_from_checkpoint
from librispeech_data import LibriSpeechDataset
from torch.utils.data import DataLoader
import whisper
import json
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import os
import numpy as np
from torch.utils.data import Dataset
from npy_append_array import NpyAppendArray

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
    
def save_activations_for_memory_mapping(activations, filenames, out_dir, layer_name):
    os.makedirs(out_dir, exist_ok=True)
    metadata_file = os.path.join(out_dir, f"{layer_name}_metadata.json")
    tensor_file = os.path.join(out_dir, f"{layer_name}_tensors.npy")
    
    # Load existing metadata if it exists
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {'filenames': [], 'tensor_shapes': []}
    
    # Prepare new data and update metadata
    new_tensors = []
    for filename, tensor in zip(filenames, activations):
        metadata['filenames'].append(filename)
        metadata['tensor_shapes'].append(list(tensor.shape))
        new_tensors.append(tensor.cpu().numpy())
    
    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    # Append tensor data to the file using NpyAppendArray
    with NpyAppendArray(tensor_file) as npaa:
        for tensor in new_tensors:
            npaa.append(tensor.reshape(1, -1))  # Reshape to 2D array for appending

def create_out_folder(folder_name):
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    return folder_name

def get_activations(
    data_path: str,
    whisper_model: torch.nn.Module,
    sae_model: Optional[AutoEncoder],
    layers_to_cache: list[str],
    split: str,
    batch_size: int,
    device: torch.device,
    out_folder_prefix: str,
):
    whisper_cache = WhisperActivationCache(
        model=whisper_model,
        activation_regex=layers_to_cache,
        device=device,
    )
    whisper_cache.model.eval()
    dataset = LibriSpeechDataset(data_path, split, device)
    # take 250-member subset for testing
    # dataset = torch.utils.data.Subset(dataset, range(250))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in tqdm(dataloader):
        whisper_cache.reset_state()
        mels, _, global_file_name, _ = batch
        with torch.no_grad():
            result = whisper_cache.forward(mels)
        activations = whisper_cache.activations
        if sae_model is not None:
            for n, act in activations.items():
                with torch.no_grad():
                    _, c = sae_model(act)
                    activations[n] = c
        for name, act in activations.items():
            out_folder = create_out_folder(f"{out_folder_prefix}/{split}/{name}")
            save_activations_for_memory_mapping(act, global_file_name, out_folder, name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="tiny.json")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    whisper_model = whisper.load_model(config["whisper_model"])
    sae_model = init_from_checkpoint(config["sae_model"]) if config["sae_model"] is not None else None
    for split in config["splits"]:
        print(f"Processing split {split}")
        get_activations(
            config["data_path"],
            whisper_model,
            sae_model,
            config["layers"],
            split,
            config["batch_size"],
            torch.device(config["device"]),
            config["out_folder_prefix"]
        )


if __name__ == "__main__":
    main()

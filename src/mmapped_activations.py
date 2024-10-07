import pathlib
import argparse
import torch
from hooked_model import WhisperActivationCache
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



class MemoryMappedActivationsDataset(Dataset):
    def __init__(self, data_dir, layer_name):
        self.data_dir = data_dir
        self.layer_name = layer_name
        self.metadata_file = os.path.join(data_dir, f"{layer_name}_metadata.json")
        self.tensor_file = os.path.join(data_dir, f"{layer_name}_tensors.npy")
        
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.mmap = np.load(self.tensor_file, mmap_mode='r')
    
    def __len__(self):
        return len(self.metadata['filenames'])
    
    def __getitem__(self, idx):
        filename = self.metadata['filenames'][idx]
        tensor_shape = self.metadata['tensor_shapes'][idx]
        tensor_offset = self.metadata['tensor_offsets'][idx]
        tensor_size = np.prod(tensor_shape)
        
        tensor_data = self.mmap[tensor_offset:tensor_offset+tensor_size].reshape(tensor_shape)
        tensor = torch.from_numpy(tensor_data)
        
        return filename, tensor

def save_activations_for_memory_mapping(activations, filenames, out_dir, layer_name):
    os.makedirs(out_dir, exist_ok=True)
    metadata = {
        'filenames': filenames,
        'tensor_shapes': [],
        'tensor_offsets': [0]
    }
    
    all_tensors = []
    for tensor in activations:
        metadata['tensor_shapes'].append(list(tensor.shape))
        all_tensors.append(tensor.numpy())
        metadata['tensor_offsets'].append(metadata['tensor_offsets'][-1] + tensor.numel())
    
    # Remove the last offset as it's not needed
    metadata['tensor_offsets'].pop()
    
    # Save metadata
    with open(os.path.join(out_dir, f"{layer_name}_metadata.json"), 'w') as f:
        json.dump(metadata, f)
    
    # Save tensors
    np.save(os.path.join(out_dir, f"{layer_name}_tensors.npy"), np.concatenate([t.flatten() for t in all_tensors]))

def create_out_folder(folder_name):
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    return folder_name

def get_activations(
    data_path: str,
    whisper_model: torch.nn.Module,
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
    # take first 500 samples
    dataset = torch.utils.data.Subset(dataset, range(500))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in tqdm(dataloader):
        whisper_cache.reset_state()
        mels, _, global_file_name, _ = batch
        with torch.no_grad():
            result = whisper_cache.forward(mels)
        activations = whisper_cache.activations
        
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
    for split in config["splits"]:
        print(f"Processing split {split}")
        get_activations(
            config["data_path"],
            whisper_model,
            config["layers"],
            split,
            config["batch_size"],
            torch.device(config["device"]),
            config["out_folder_prefix"]
        )


if __name__ == "__main__":
    main()

import argparse
import json
import os
import pathlib
import torch
from tqdm import tqdm
from typing import Optional
from src.dataset.activation_dataset import FlyActivationDataloader
from npy_append_array import NpyAppendArray

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
    layer_to_cache: str,
    whisper_model: str,
    batch_size: int,
    device: torch.device,
    out_folder: str,
    collect_max: Optional[int]
):
    dataloader = FlyActivationDataloader(
        data_path,
        whisper_model,
        None,
        layer_to_cache,
        device,
        batch_size,
        4,
        collect_max
    )
    for batch in tqdm(dataloader):
        activations, global_filenames = batch
        save_activations_for_memory_mapping(activations, global_filenames, out_folder, layer_to_cache)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
        get_activations(
            config["data_path"],
            config["layer_to_cache"],
            config["whisper_model"],
            config["batch_size"],
            torch.device(config["device"]),
            create_out_folder(config["out_folder"]),
            config["collect_max"]
        )

if __name__ == "__main__":
    main()

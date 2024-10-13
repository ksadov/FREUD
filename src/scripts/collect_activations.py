import argparse
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List
from npy_append_array import NpyAppendArray
from src.dataset.activations import FlyActivationDataLoader


def save_data_for_memory_mapping(metadata_file: Path, data_files: List[Path], data: List[torch.Tensor],
                                 filenames: List[str], tensor_shape: List[int], activation_shape: List[int]):
    """
    Append data to memory-mappable file(s) and update metadata

    :param metadata_file: Path to the metadata file
    :param data_files: List of paths to the data file(s)
    :param data: List of data tensors to save
    :param filenames: List of filenames corresponding to the data
    """
    assert len(data[0]) == len(
        filenames), "Number of data tensors and filenames must match"

    # Load or initialize metadata
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {
            'tensor_shape': tensor_shape,
            'activation_shape': activation_shape,
            'filenames': []
        }

    # Update metadata and prepare new data
    new_data = [[] for _ in data]
    for filename, *tensors in zip(filenames, *data):
        metadata['filenames'].append(filename)
        for i, tensor in enumerate(tensors):
            if metadata['tensor_shape'] != list(tensor.shape):
                raise ValueError(
                    f"All tensors must have the same shape as the first tensor. Expected {metadata['tensor_shape'][i]}, got {tensor.shape}")
            new_data[i].append(tensor.cpu().numpy())

    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

    # Append data to file(s) using NpyAppendArray
    for file, tensors in zip(data_files, new_data):
        with NpyAppendArray(file) as npaa:
            for tensor in tensors:
                npaa.append(tensor.reshape(1, -1))


def get_activations(
    data_path: str,
    layer_name: str,
    whisper_model: str,
    sae_model: Optional[str],
    batch_size: int,
    device: torch.device,
    out_folder: str,
    max_workers: int,
    collect_max: Optional[int]
):
    """
    Collect activations from whisper_model or sae_model

    :param data_path: Path to the data folder
    :param layer_name: Layer to cache activations for
    :param whisper_model: String corresponding to the whisper model name, i.e "tiny"
    :param sae_model: Path to SAE checkpoint
    :param batch_size: Batch size for the dataloader
    :param device: Device to run the model on
    :param out_folder: Output folder for saving data
    :param max_workers: Maximum number of workers for dataloader
    :param collect_max: Maximum number of samples to collect (optional)
    """
    dataloader = FlyActivationDataLoader(
        data_path, whisper_model, sae_model, layer_name, device,
        batch_size, max_workers, collect_max
    )

    metadata_file = Path(out_folder) / f"{layer_name}_metadata.json"
    data_files = [Path(out_folder) / f"{layer_name}_tensors.npy"]

    if dataloader.activation_type != "tensor":
        data_files = [
            Path(out_folder) / f"{layer_name}_activation_values.npy",
            Path(out_folder) / f"{layer_name}_feature_indices.npy"
        ]

    # Remove existing files
    for file in [metadata_file] + data_files:
        if file.exists():
            file.unlink()

    # Create output directory
    Path(out_folder).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if dataloader.activation_type == "tensor":
                activation, global_filenames = batch
                data = [activation]
                tensor_shape = list(activation[0].shape)
            else:
                act_data, index_data, global_filenames = batch
                data = [act_data, index_data]
                tensor_shape = list(act_data[0].shape)

            save_data_for_memory_mapping(
                metadata_file, data_files, data, global_filenames, tensor_shape, dataloader.activation_shape
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        help="Path to feature configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    get_activations(
        config["data_path"],
        config["layer_name"],
        config["whisper_model"],
        config.get("sae_model"),
        config["batch_size"],
        config["device"],
        config["out_folder"],
        config["dl_max_workers"],
        config.get("collect_max")
    )


if __name__ == "__main__":
    main()

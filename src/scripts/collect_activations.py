import argparse
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from npy_append_array import NpyAppendArray

from src.dataset.activations import FlyActivationDataLoader


def save_activations_for_memory_mapping(metadata_file: Path, tensor_file: Path, activations: torch.tensor,
                                        filenames: list[str]):
    """
    Append activations to a memory-mappable file and update metadata

    :param metadata_file: Path to the metadata file
    :param tensor_file: Path to the tensor file
    :param activations: List of activations to save
    :param filenames: List of filenames corresponding to the activations
    :requires: len(activations) == len(filenames)
    """
    assert len(activations) == len(
        filenames), "Number of activations and filenames must match"
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
            # Reshape to 2D array for appending
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
    If sae_model is specified, collect activations from sae_model, otherwise collect activations from whisper_model

    :param data_path: Path to the data folder
    :param layer_name: Layer to cache activations for
    :param whisper_model: String corresponding to the whisper model name, i.e "tiny"
    :param sae_model: Path to SAE checkpoint
    :param batch_size: Batch size for the dataloader
    :param device: Device to run the model on
    """
    dataloader = FlyActivationDataLoader(
        data_path,
        whisper_model,
        sae_model,
        layer_name,
        device,
        batch_size,
        max_workers,
        collect_max
    )

    metadata_file = os.path.join(out_folder, f"{layer_name}_metadata.json")
    tensor_file = os.path.join(out_folder, f"{layer_name}_tensors.npy")
    # delete existing metadata and tensor files
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    if os.path.exists(tensor_file):
        os.remove(tensor_file)
    # create directory for the output
    os.makedirs(out_folder, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            activations, global_filenames = batch
            save_activations_for_memory_mapping(
                metadata_file, tensor_file, activations, global_filenames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        help="Path to feature configuration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
        os.makedirs(config["out_folder"], exist_ok=True)
        get_activations(
            config["data_path"],
            config["layer_name"],
            config["whisper_model"],
            config["sae_model"],
            config["batch_size"],
            config["device"],
            config["out_folder"],
            config["dl_max_workers"],
            config.get("collect_max", None)
        )


if __name__ == "__main__":
    main()

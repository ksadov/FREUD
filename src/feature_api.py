import os
from typing import Optional
import torch
import torchaudio
from constants import SAMPLE_RATE, TIMESTEP_S
from mmapped_activations import MemoryMappedActivationsDataset
from activation_dataset import FlyActivationDataloader
from tqdm import tqdm


def load_activations(batch_folder: str) -> dict:
    activation_map = {}
    for batch_file in os.listdir(batch_folder)[:50]:
        batch_path = os.path.join(batch_folder, batch_file)
        batch = torch.load(batch_path)
        # if batch values are tuples, take the first element
        values_are_tuples = isinstance(list(batch.items())[0][1], tuple)
        if values_are_tuples:
            batch = {k: v[0] for k, v in batch.items()}
        activation_map.update(batch)
    return activation_map


def init_map(layer_name: str, config: dict, split: str, files_to_search: int) -> torch.Tensor:
    data_dir = config['out_folder']
    dset = MemoryMappedActivationsDataset(data_dir, layer_name, files_to_search)
    print("dset len", len(dset))
    return dset


def trim_activation(audio_fname: str, activation: torch.Tensor) -> torch.Tensor:
    audio_duration = torchaudio.info(audio_fname).num_frames / SAMPLE_RATE
    n_frames = int(audio_duration / TIMESTEP_S)
    return activation[:n_frames]


def get_top_activating_files(activation_audio_map: dict, n_files: int, neuron_idx: int) -> list[str]:
    top_files = top_activating_files(activation_audio_map, n_files, neuron_idx)
    return [x[1] for x in top_files[:n_files]]


def top_activating_files(activation_audios: MemoryMappedActivationsDataset, n_files: int, neuron_idx: int, max_val: Optional[float]) -> list:
    top = []
    for audio_file, activation in activation_audios:
        activation_at_idx = activation.transpose(0, 1)[neuron_idx]
        trimmed_activation = trim_activation(audio_file, activation_at_idx)
        max_activation_value = trimmed_activation.max().item()
        if max_val is None or max_activation_value < max_val:
            max_activation_loc = trimmed_activation.argmax().item()
            max_activation_time = max_activation_loc * TIMESTEP_S
            top.append((audio_file, trimmed_activation,
                        max_activation_value, max_activation_time))
            top.sort(key=lambda x: x[2], reverse=True)
            top = top[:n_files]
    return top

def search_activations(batch_folder, neuron_idx, n_files, max_val):
    # activation map may be too big to load all at once
    # so we just dynamically load the activations for the batch
    # and return the top n files
    print("Searching activations...")
    print(f"Neuron index: {neuron_idx}")
    print(f"N files: {n_files}")
    top = []
    for batch_file in os.listdir(batch_folder):
        batch_path = os.path.join(batch_folder, batch_file)
        batch = torch.load(batch_path)
        top_batch_files = top_activating_files(batch, n_files, neuron_idx, max_val)
        if len(top) < n_files:
            top.extend(top_batch_files)
        else:
            top.extend(top_batch_files)
            top.sort(key=lambda x: x[2], reverse=True)
            top = top[:n_files]
    return top

def get_top_fly(dataloader: FlyActivationDataloader, neuron_idx: int, n_files: int, max_val: Optional[float]) -> list:
    print("Searching activations...")
    pq = []
    for audio_files, act_batch in tqdm(dataloader):
        for act, audio_file in zip(act_batch, audio_files):
            act = act.squeeze()[:, neuron_idx]
            trimmed_activation = trim_activation(audio_file, act)
            max_activation_value = trimmed_activation.max().item()
            if max_val is None or max_activation_value < max_val:
                max_activation_loc = trimmed_activation.argmax().item()
                max_activation_time = max_activation_loc * TIMESTEP_S
                pq.append((audio_file, trimmed_activation, max_activation_value, max_activation_time))
                pq.sort(key=lambda x: x[2], reverse=True)
                pq = pq[:n_files]
    print("Search complete.")
    return pq

def make_top_fn(config: dict, layer_name: str, split: str, from_disk: bool, files_to_search: Optional[int]) -> callable:
    if from_disk:
        activation_audio_map = init_map(layer_name, config, split, files_to_search)
        return lambda neuron_idx, n_files, max_val: top_activating_files(
            activation_audio_map, n_files, neuron_idx, max_val)
    else:
        fly_dataloader = FlyActivationDataloader(
            config['data_path'],
            config['whisper_model'],
            config['sae_model'],
            layer_name,
            config['device'],
            split,
            batch_size=50,
            dl_max_workers=4,
            subset_size=files_to_search
        )
        return lambda neuron_idx, n_files, max_val: get_top_fly(
            fly_dataloader, neuron_idx, n_files, max_val)
        
def get_top_activations(top_fn: callable,
                        neuron_idx: int,
                        n_files: int,
                        max_val: Optional[float] = None
                        ) -> tuple[list[str], list[torch.Tensor]]:
    top = top_fn(neuron_idx, n_files, max_val)
    top_files = [x[0] for x in top]
    activations = [x[1] for x in top]
    print("Got top activations.")
    return top_files, activations

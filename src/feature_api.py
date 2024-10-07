import os
from typing import Optional
import torch
import torchaudio
from librispeech_data import LibriSpeechDataset
from constants import SAMPLE_RATE, TIMESTEP_S
from autoencoder import init_from_checkpoint, AutoEncoder, get_audio_features
from hooked_model import init_cache, WhisperActivationCache
from librispeech_data import get_librispeech_files
from typing import Generator


def get_batch_folder(config: dict, split: str, layer_name: str) -> str:
    batch_folder = f"{config['out_folder_prefix']}/{split}/{layer_name}"
    return batch_folder


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


def init_map(layer_name: str, config: dict, split: str) -> torch.Tensor:
    batch_folder = get_batch_folder(config, split, layer_name)
    activation_map = load_activations(batch_folder)
    activation_audio_map = activation_map
    return activation_audio_map


def trim_activation(audio_fname: str, activation: torch.Tensor) -> torch.Tensor:
    audio_duration = torchaudio.info(audio_fname).num_frames / SAMPLE_RATE
    n_frames = int(audio_duration / TIMESTEP_S)
    return activation[:n_frames]


def get_activation(neuron_idx: int, audio_fname: str, activation_audio_map: dict) -> list:
    if audio_fname in activation_audio_map:
        activation = activation_audio_map[audio_fname]
        return activation.transpose(0, 1)[neuron_idx]
    else:
        raise ValueError(
            f"Audio file {audio_fname} not found in activation_audio_map")


def get_top_activating_files(activation_audio_map: dict, n_files: int, neuron_idx: int) -> list[str]:
    top_files = top_activating_files(activation_audio_map, n_files, neuron_idx)
    return [x[1] for x in top_files[:n_files]]


def top_activating_files(activation_audio_map: dict, n_files: int, neuron_idx: int, max_val: Optional[float]) -> list:
    top = []
    for audio_file, activation in activation_audio_map.items():
        activation_at_idx = activation.transpose(0, 1)[neuron_idx]
        trimmed_activation = trim_activation(audio_file, activation_at_idx)
        max_activation_value = trimmed_activation.max().item()
        if max_val is None or max_activation_value < max_val:
            max_activation_loc = trimmed_activation.argmax().item()
            max_activation_time = max_activation_loc * TIMESTEP_S
            top.append((audio_file, trimmed_activation,
                        max_activation_value, max_activation_time))
    top.sort(key=lambda x: x[2], reverse=True)
    return top[:n_files]

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

def get_top_sae(sae_model: AutoEncoder, whisper_cache: WhisperActivationCache, audio_files: Generator[str, None, None], 
                neuron_idx: int, n_files: int, max_val: Optional[float]) -> list:
    print("Searching SAE activations...")
    # too big to keep all in memory, make a priority queue
    pq = []
    for i, audio_path in enumerate(audio_files):
        if i > 10:
            break
        audio_features = get_audio_features(sae_model, whisper_cache, audio_path)
        activation = audio_features[:, :, neuron_idx].squeeze()
        print("ACTIVATION SHAPE", activation.shape)
        max_activation_value = activation.max().item()
        if max_val is None or max_activation_value < max_val:
            max_activation_loc = activation.argmax().item()
            max_activation_time = max_activation_loc * TIMESTEP_S
            pq.append((audio_path, activation, max_activation_value, max_activation_time))
            pq.sort(key=lambda x: x[2], reverse=True)
            pq = pq[:n_files]
    return pq

def make_top_fn(config: dict, layer_name: str, split: str, init_at_start: bool) -> callable:
    if init_at_start:
        activation_audio_map = init_map(layer_name, config, split)
        return lambda neuron_idx, n_files, max_val: top_activating_files(
            activation_audio_map, n_files, neuron_idx, max_val)
    elif config['model_type'] == 'sae':
        sae_model = init_from_checkpoint(config['sae_model'])
        whisper_cache = init_cache(config['whisper_model'], layer_name, device=config['device'])
        print("models loaded")
        audio_files = get_librispeech_files(config['data_path'], split)
        return lambda neuron_idx, n_files, max_val: get_top_sae(
            sae_model, whisper_cache, audio_files, neuron_idx, n_files, max_val
        )
    elif config['model_type'] == 'whisper':
        batch_dir = get_batch_folder(config, split, layer_name)
        return lambda neuron_idx, n_files, max_val: search_activations(
            batch_dir, neuron_idx, n_files, max_val)
    else:
        raise ValueError(f"Invalid model type {config['model_type']}, must be 'sae' or 'whisper'.")
        
def get_top_activations(top_fn: callable,
                        neuron_idx: int,
                        n_files: int,
                        max_val: Optional[float] = None
                        ) -> tuple[list[str], list[torch.Tensor]]:
    top = top_fn(neuron_idx, n_files, max_val)
    top_files = [x[0] for x in top]
    activations = [x[1] for x in top]
    return top_files, activations

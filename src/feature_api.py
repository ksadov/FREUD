import os
import torch
import torchaudio
from librispeech_data import LibriSpeechDataset
from constants import SAMPLE_RATE, TIMESTEP_S


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
    batch_folder = f"{config['out_folder_prefix']}/{split}/{layer_name}"
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


def top_activating_files(activation_audio_map: dict, n_files: int, neuron_idx: int) -> list:
    top = []
    for audio_file, activation in activation_audio_map.items():
        activation_at_idx = activation.transpose(0, 1)[neuron_idx]
        trimmed_activation = trim_activation(audio_file, activation_at_idx)
        max_activation_value = trimmed_activation.max().item()
        max_activation_loc = trimmed_activation.argmax().item()
        max_activation_time = max_activation_loc * TIMESTEP_S
        top.append((audio_file, trimmed_activation,
                    max_activation_value, max_activation_time))
    top.sort(key=lambda x: x[2], reverse=True)
    return top[:n_files]


def get_top_activations(activation_audio_map, n_files, neuron_idx) -> tuple[list[str], list[torch.Tensor]]:
    top = top_activating_files(
        activation_audio_map, n_files, neuron_idx)
    print("top", top)
    top_files = [x[0] for x in top]
    activations = [x[1] for x in top]
    return top_files, activations

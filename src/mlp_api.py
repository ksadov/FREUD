import torch
import torchaudio
from librispeech_data import LibriSpeechDataset
from activation_data import load_activations, id_audio_assoc, activation_audio_assoc
from constants import SAMPLE_RATE, TIMESTEP_S


def init_map(layer_name: str, config: dict, split: str) -> torch.Tensor:
    batch_folder = f"{config['out_folder_prefix']}/{split}/{layer_name}"
    activation_map = load_activations(batch_folder)
    dataset = LibriSpeechDataset(
        config["data_path"], split, torch.device(config["device"]))
    id_audio_map = id_audio_assoc(dataset)
    activation_audio_map = activation_audio_assoc(activation_map, id_audio_map)
    # convert to dict where keys are tensors and values are strings
    activation_audio_map = {k: v for k, v in activation_audio_map.items()}
    return activation_audio_map


def trim_activation(audio_fname: str, activation: torch.Tensor) -> torch.Tensor:
    audio_duration = torchaudio.info(audio_fname).num_frames / SAMPLE_RATE
    n_frames = int(audio_duration / TIMESTEP_S)
    return activation[:n_frames]


def get_activation(neuron_idx: int, audio_fname: str, activation_audio_map: dict) -> list:
    for activation, audio_file in activation_audio_map.items():
        if audio_file == audio_fname:
            return trim_activation(audio_fname, activation.transpose(0, 1)[neuron_idx]).tolist()
    raise ValueError(
        f"Audio file {audio_fname} not found in activation_audio_map")

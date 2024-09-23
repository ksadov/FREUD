import os
import torch
import torchaudio
from librispeech_data import LibriSpeechDataset
from constants import SAMPLE_RATE, TIMESTEP_S


def load_activations(batch_folder: str) -> dict:
    activation_map = {}
    for batch_file in os.listdir(batch_folder)[:50]:
        batch_path = os.path.join(batch_folder, batch_file)
        activation_map.update(torch.load(batch_path))
    return activation_map


def id_audio_assoc(dataset: LibriSpeechDataset) -> dict:
    id_audio_map = {}
    for i in range(len(dataset)):
        _, utterance_id, audio_file, transcript = dataset[i]
        id_audio_map[utterance_id] = (audio_file, transcript)
    return id_audio_map


def activation_audio_assoc(activation_map: dict, id_audio_map: dict) -> dict:
    activation_audio_map = {}
    print("ID AUDIO MAP KEYS: ", id_audio_map.keys())
    print("ACTIVATION MAP KEYS: ", activation_map.keys())
    for utterance_id, activation in activation_map.items():
        utterance_int = utterance_id
        audio_file = id_audio_map[utterance_int]
        print("ACTIVATION for utterance_id: ",
              utterance_id, ": ", activation[1])
        print("TRANSCRIPT for utterance_id: ",
              utterance_id, ": ", audio_file[1])
        activation_audio_map[audio_file] = activation
    return activation_audio_map


def init_map(layer_name: str, config: dict, split: str) -> torch.Tensor:
    batch_folder = f"{config['out_folder_prefix']}/{split}/{layer_name}"
    activation_map = load_activations(batch_folder)
    """
    dataset = LibriSpeechDataset(
        config["data_path"], split, torch.device(config["device"]), calculate_mel=False)
    id_audio_map = id_audio_assoc(dataset)
    activation_audio_map = activation_audio_assoc(activation_map, id_audio_map)
    activation_audio_map = {k: v for k, v in activation_audio_map.items()}
    """
    activation_audio_map = activation_map
    return activation_audio_map


def trim_activation(audio_fname: str, activation: torch.Tensor) -> torch.Tensor:
    audio_duration = torchaudio.info(audio_fname).num_frames / SAMPLE_RATE
    n_frames = int(audio_duration / TIMESTEP_S)
    return activation[:n_frames]


def get_activation(neuron_idx: int, audio_fname: str, activation_audio_map: dict) -> list:
    if audio_fname in activation_audio_map:
        activation = activation_audio_map[audio_fname]
        return activation[0].transpose(0, 1)[neuron_idx]
    else:
        raise ValueError(
            f"Audio file {audio_fname} not found in activation_audio_map")


def get_top_activating_files(activation_audio_map: dict, n_files: int, neuron_idx: int) -> list[str]:
    print("GETTING TOP FILES")
    top_files = top_activating_files(activation_audio_map, n_files, neuron_idx)
    return [x[1] for x in top_files[:n_files]]


def top_activating_files(activation_audio_map: dict, n_files: int, neuron_idx: int) -> list:
    top = []
    for audio_file, activation in activation_audio_map.items():
        activation_at_idx = activation[0].transpose(0, 1)[neuron_idx]
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

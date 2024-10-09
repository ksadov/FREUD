import os
from typing import Optional
import torch
import torchaudio
from src.utils.constants import SAMPLE_RATE, TIMESTEP_S
from src.dataset.activations import MemoryMappedActivationsDataset, FlyActivationDataloader
from tqdm import tqdm

def trim_activation(audio_fname: str, activation: torch.Tensor) -> torch.Tensor:
    """
    Trim the activation tensor to match the duration of the audio file
    """
    audio_duration = torchaudio.info(audio_fname).num_frames / SAMPLE_RATE
    n_frames = int(audio_duration / TIMESTEP_S)
    return activation[:n_frames]

def mmapped_top(activation_audios: MemoryMappedActivationsDataset, neuron_idx: int, n_files: int, 
                         max_val: Optional[float]) -> list:
    """
    Given a memory-mapped activation dataset, return the n files that activate a given neuron the most, or the top n 
    files under a given maximum activation value

    :param activation_audios: MemoryMappedActivationsDataset
    :param neuron_idx: index of the neuron to search for
    :param n_files: number of files to return
    :param max_val: maximum activation value to consider, or None to find the files with the highest activations
    """
    top = []
    print("Searching activations...")
    for activation, audio_file in tqdm(activation_audios):
        activation_at_idx = activation.transpose(0, 1)[neuron_idx]
        trimmed_activation = trim_activation(audio_file, activation_at_idx)
        max_activation_value = trimmed_activation.max().item()
        if max_val is None or max_activation_value < max_val:
            max_activation_loc = trimmed_activation.argmax().item()
            max_activation_time = max_activation_loc * TIMESTEP_S
            top.append((audio_file, trimmed_activation, max_activation_value, max_activation_time))
            top.sort(key=lambda x: x[2], reverse=True)
            top = top[:n_files]
    print("Search complete.")
    return top

def fly_top(dataloader: FlyActivationDataloader, neuron_idx: int, n_files: int, max_val: Optional[float]) -> list:
    """
    Given a dataloader that generates activations on the fly, return the n files that activate a given neuron the most, 
    or the top n files under a given maximum activation value

    :param dataloader: FlyActivationDataloader
    :param neuron_idx: index of the neuron to search for
    :param n_files: number of files to return
    :param max_val: maximum activation value to consider, or None to find the files with the highest activations
    """
    print("Searching activations...")
    pq = []
    for act_batch, audio_files in tqdm(dataloader):
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

def make_top_fn(config: dict, from_disk: bool, files_to_search: Optional[int]) -> callable:
    if from_disk:
        activation_audio_map = MemoryMappedActivationsDataset(
            config['out_folder'],
            config['layer_to_cache'],
            max_size=files_to_search
        )
        return lambda neuron_idx, n_files, max_val: mmapped_top(
            activation_audio_map, neuron_idx, n_files, max_val
        )
    else:
        fly_dataloader = FlyActivationDataloader(
            config['data_path'],
            config['whisper_model'],
            config['sae_model'],
            config['layer_to_cache'],
            config['device'],
            config['batch_size'],
            dl_max_workers=4,
            subset_size=files_to_search
        )
        return lambda neuron_idx, n_files, max_val: fly_top(
            fly_dataloader, neuron_idx, n_files, max_val
        )
        
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

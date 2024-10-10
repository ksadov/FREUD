import os
from typing import Optional
import torch
import torchaudio
from src.utils.constants import SAMPLE_RATE, TIMESTEP_S
from src.dataset.activations import MemoryMappedActivationDataLoader, FlyActivationDataLoader
from tqdm import tqdm

def trim_activation(audio_fname: str, activation: torch.Tensor) -> torch.Tensor:
    """
    Trim the activation tensor to match the duration of the audio file
    """
    audio_duration = torchaudio.info(audio_fname).num_frames / SAMPLE_RATE
    n_frames = int(audio_duration / TIMESTEP_S)
    return activation[:n_frames]


def top_activations(dataloader: MemoryMappedActivationDataLoader | FlyActivationDataLoader, neuron_idx: int, n_files: int, max_val: Optional[float]) -> list:
    """
    Given an activation dataloader, return the n files that activate a given neuron the most, 
    or the top n files under a given maximum activation value

    :param dataloader: MemoryMappedActivationDataLoader | FlyActivationDataLoader
    :param neuron_idx: index of the neuron to search for
    :param n_files: number of files to return
    :param max_val: maximum activation value to consider, or None to find the files with the highest activations
    """
    print("Searching activations...")
    pq = []
    for act_batch, audio_files in tqdm(dataloader):
        for act, audio_file in zip(act_batch, audio_files):
            act = act[:, neuron_idx]
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
        dataloader = MemoryMappedActivationDataLoader(
            config['out_folder'],
            config['layer_name'],
            config['batch_size'],
            dl_max_workers=config['dl_max_workers'],
            subset_size=files_to_search
        )
    else:
        dataloader = FlyActivationDataLoader(
            config['data_path'],
            config['whisper_model'],
            config['sae_model'],
            config['layer_name'],
            config['device'],
            config['batch_size'],
            dl_max_workers=config['dl_max_workers'],
            subset_size=files_to_search
        )
    return lambda neuron_idx, n_files, max_val: top_activations(dataloader, neuron_idx, n_files, max_val)
        
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

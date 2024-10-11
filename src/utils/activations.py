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


def top_activations(dataloader: MemoryMappedActivationDataLoader | FlyActivationDataLoader, neuron_idx: int, 
                    n_files: int, max_val: Optional[float], min_val: Optional[float], 
                    absolute_magnitude: bool) -> list:
    """
    Given an activation dataloader, return the n files that activate a given neuron the most within an optional range of values.

    :param dataloader: MemoryMappedActivationDataLoader | FlyActivationDataLoader
    :param neuron_idx: index of the neuron to search for
    :param n_files: number of files to return
    :param max_val: maximum activation value to consider, or None
    :param min_val: minimum activation value to consider, or None
    :param absolute_magnitude: search for the top n files with the highest absolute magnitude activations
    """
    print("Searching activations...")
    pq = []
    for act_batch, audio_files in tqdm(dataloader):
        for act, audio_file in zip(act_batch, audio_files):
            act = act[:, neuron_idx]
            trimmed_activation = trim_activation(audio_file, act)
            def filter_activation(max_activation_value: torch.Tensor) -> bool:
                if max_val is not None and max_activation_value > max_val:
                    return False
                if min_val is not None and max_activation_value < min_val:
                    return False
                return True
            if absolute_magnitude:
                max_activation_index = torch.argmax(torch.abs(trimmed_activation))
                signed_max_activation_value = trimmed_activation[max_activation_index].item()
                allow_activation = filter_activation(signed_max_activation_value)
                max_activation_value = abs(signed_max_activation_value)
            else:
                max_activation_value = trimmed_activation.max().item()
                allow_activation = filter_activation(max_activation_value)
            if allow_activation:
                max_activation_loc = trimmed_activation.argmax().item()
                max_activation_time = max_activation_loc * TIMESTEP_S
                pq.append((audio_file, trimmed_activation, max_activation_value, max_activation_time))
                pq.sort(key=lambda x: x[2], reverse=True)
                pq = pq[:n_files]
    print("Search complete.")
    return pq
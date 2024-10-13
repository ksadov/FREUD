from typing import Optional
import torch
import torchaudio
from tqdm import tqdm

from src.utils.constants import SAMPLE_RATE, TIMESTEP_S
from src.dataset.activations import MemoryMappedActivationDataLoader, FlyActivationDataLoader
from src.models.l1autoencoder import L1EncoderOutput
from src.models.topkautoencoder import TopKEncoderOutput


def trim_activation(audio_fname: str, activation: torch.Tensor) -> torch.Tensor:
    """
    Trim the activation tensor to match the duration of the audio file
    """
    audio_duration = torchaudio.info(audio_fname).num_frames / SAMPLE_RATE
    n_frames = int(audio_duration / TIMESTEP_S)
    return activation[:n_frames]


def activation_tensor_from_indexed(activation_values: torch.Tensor, activation_indices: torch.Tensor, neuron_idx: int) -> torch.Tensor:
    """
    Convert an indexed activation tensor to a dense tensor
    """
    act = torch.zeros(len(activation_indices), activation_indices.shape[1])
    for i, top_indices_per_file in enumerate(activation_indices):
        neuron_act = torch.zeros(len(top_indices_per_file))
        for j, top_indices_per_timestep in enumerate(top_indices_per_file):
            if neuron_idx in top_indices_per_timestep:
                index_of_neuron = (top_indices_per_timestep == neuron_idx).nonzero(
                ).item()
                neuron_act[j] = activation_values[i][j][index_of_neuron]
        act[i] = neuron_act
    return act


@torch.no_grad()
def top_activations(dataloader: MemoryMappedActivationDataLoader | FlyActivationDataLoader, neuron_idx: int,
                    n_files: int, max_val: Optional[float], min_val: Optional[float],
                    absolute_magnitude: bool, return_max_per_file: bool) -> \
        tuple[list[tuple[str, torch.Tensor, float, float]], list[float], Optional[list[float]]]:
    """
    Given an activation dataloader, return the n files that activate a given neuron the most within an optional range of values.

    :param dataloader: MemoryMappedActivationDataLoader | FlyActivationDataLoader
    :param neuron_idx: index of the neuron to search for
    :param n_files: number of files to return
    :param max_val: maximum activation value to consider, or None
    :param min_val: minimum activation value to consider, or None
    :param absolute_magnitude: search for the top n files with the highest absolute magnitude activations
    :param return_all_maxes: return a list of the max activation for each file
    :return: list of tuples (audio_file, trimmed_activation, max_activation_value, max_activation_time) and optionally a list of max activations
    """
    print("Searching activations...")
    pq = []
    max_per_file = []

    def filter_activation(max_activation_value: torch.Tensor) -> bool:
        if max_val is not None and max_activation_value > max_val:
            return False
        if min_val is not None and max_activation_value < min_val:
            return False
        return True

    for batch in tqdm(dataloader):
        if dataloader.activation_type == "tensor":
            act_batch, audio_files = batch
            act = act_batch[:, :, neuron_idx]
        else:
            act_batch, indexes, audio_files = batch
            act = activation_tensor_from_indexed(
                act_batch, indexes, neuron_idx)
        for audio_file, act in zip(audio_files, act):
            trimmed_activation = trim_activation(audio_file, act)
            if absolute_magnitude:
                max_activation_index = torch.argmax(
                    torch.abs(trimmed_activation))
                signed_max_activation_value = trimmed_activation[max_activation_index].item(
                )
                allow_activation = filter_activation(
                    signed_max_activation_value)
                max_activation_value = abs(signed_max_activation_value)
                if return_max_per_file:
                    max_per_file.append(signed_max_activation_value)
            else:
                max_activation_value = trimmed_activation.max().item()
                allow_activation = filter_activation(max_activation_value)
                if return_max_per_file:
                    max_per_file.append(max_activation_value)
            if allow_activation:
                max_activation_loc = trimmed_activation.argmax().item()
                max_activation_time = max_activation_loc * TIMESTEP_S
                pq.append((audio_file, trimmed_activation,
                           max_activation_value, max_activation_time))
                pq.sort(key=lambda x: x[2], reverse=True)
                pq = pq[:n_files]
    print("Search complete.")
    return pq, None if not return_max_per_file else max_per_file

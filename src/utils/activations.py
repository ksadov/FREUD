from typing import Optional
import torch
import torchaudio
from tqdm import tqdm
import numpy as np

from src.utils.constants import SAMPLE_RATE, TIMESTEP_S
from src.dataset.activations import MemoryMappedActivationDataLoader, FlyActivationDataLoader
from src.models.l1autoencoder import L1EncoderOutput, L1AutoEncoder
from src.models.topkautoencoder import TopKAutoEncoder
from src.models.hooked_model import WhisperActivationCache, WhisperSubbedActivation
from src.utils.audio_utils import get_mels_from_np_array


def trim_activation(audio_fname: str, activation: torch.Tensor) -> torch.Tensor:
    """
    Trim the activation tensor to match the duration of the audio file
    """
    audio_duration = torchaudio.info(audio_fname).num_frames / SAMPLE_RATE
    n_frames = int(audio_duration / TIMESTEP_S)
    return activation[:n_frames]


def activation_length_from_audio_array(audio_array: np.ndarray) -> torch.Tensor:
    """
    Get the number of frames in the activation tensor from an audio array
    """
    audio_duration = len(audio_array) / SAMPLE_RATE
    n_frames = int(audio_duration / TIMESTEP_S)
    return n_frames


def activation_tensor_from_indexed(activation_values: torch.Tensor, activation_indices: torch.Tensor, feature_idx: int) -> torch.Tensor:
    """
    Convert an indexed activation tensor to a dense tensor
    """
    act = torch.zeros(len(activation_indices), activation_indices.shape[1])
    for i, top_indices_per_file in enumerate(activation_indices):
        feature_act = torch.zeros(len(top_indices_per_file))
        for j, top_indices_per_timestep in enumerate(top_indices_per_file):
            if feature_idx in top_indices_per_timestep:
                index_of_feature = (top_indices_per_timestep == feature_idx).nonzero(
                ).item()
                feature_act[j] = activation_values[i][j][index_of_feature]
        act[i] = feature_act
    return act


@torch.no_grad()
def top_activations(dataloader: MemoryMappedActivationDataLoader | FlyActivationDataLoader, feature_idx: int,
                    n_files: int, max_val: Optional[float], min_val: Optional[float],
                    absolute_magnitude: bool, return_max_per_file: bool) -> \
        tuple[list[tuple[str, torch.Tensor, float, float]], list[float], Optional[list[float]]]:
    """
    Given an activation dataloader, return the n files that activate a given feature the most within an optional range of values.

    :param dataloader: MemoryMappedActivationDataLoader | FlyActivationDataLoader
    :param feature_idx: index of the feature to search for
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
            act = act_batch[:, :, feature_idx]
        else:
            act_batch, indexes, audio_files = batch
            act = activation_tensor_from_indexed(
                act_batch, indexes, feature_idx)
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


@torch.no_grad()
def top_activations_for_audio(audio_array: np.ndarray, whisper_cache: WhisperActivationCache,
                              sae_model: Optional[L1AutoEncoder | TopKAutoEncoder],
                              top_n: int) -> tuple[list[int], list[float]]:
    """
    Given input audio, get the top features encoded by the whisper model and optionally the SAE model.

    :param audio_array: array of audio samples
    :param whisper_model: Whisper model
    :param sae_model: SAE model
    :param top_n: Number of top features to return
    :return: Tuple of top feature indices and their corresponding values
    :requires: audio_array must be sampled at 16kHz
    """

    mel = get_mels_from_np_array(whisper_cache.device, audio_array)
    whisper_cache.forward(mel)
    activations = whisper_cache.activations
    indexed_activations = False
    true_length = activation_length_from_audio_array(audio_array)
    if sae_model:
        output = sae_model.forward(activations)
        if isinstance(output.encoded, L1EncoderOutput):
            activations = output.encoded.latent
        else:
            top_acts = output.encoded.top_acts.squeeze()[:true_length, :]
            top_indices = output.encoded.top_indices.squeeze()[:true_length, :]
            indexed_activations = True

    if not indexed_activations:
        activations = activations.squeeze()
        activations = activations[:true_length, :]
        top_k_results = activations.topk(top_n)
        top_acts, top_indices = top_k_results.values, top_k_results.indices

    unique_top_activations = []
    for top_acts_at_t, top_indices_at_t in zip(top_acts, top_indices):
        unique_top_activations.extend(
            [(idx.item(), value.item()) for idx, value in zip(top_indices_at_t, top_acts_at_t)])
        # sort by value
        unique_top_activations = sorted(
            unique_top_activations, key=lambda x: x[1], reverse=True)
        new_unique = []
        for i, (idx, value) in enumerate(unique_top_activations):
            if idx not in [idx for idx, _ in new_unique] and len(new_unique) < top_n:
                new_unique.append((idx, value))
        unique_top_activations = new_unique

    # sanity check
    max_activations = []
    for i, v in unique_top_activations:
        if indexed_activations:
            act = activation_tensor_from_indexed(
                top_acts.unsqueeze(0), top_indices.unsqueeze(0), i)
            print("ACT SHAPE", act.shape)
        else:
            act = activations[:, i]
            print("ACT SHAPE", act.shape)
        assert act.max(
        ) == v, f"Max activation at index {i} is {act.max()} but expected {v}"
        max_activations.append(act)
    activation_indexes = [i for i, _ in unique_top_activations]
    return activation_indexes, max_activations


def manipulate_latent(audio_array: np.ndarray, whisper_cache: WhisperActivationCache,
                      sae_model: Optional[L1AutoEncoder | TopKAutoEncoder],
                      whisper_subbed: WhisperSubbedActivation, feat_idx: int,
                      manipulation_factor: float) -> tuple[Optional[str], str, str, torch.Tensor, torch.Tensor]:
    """
    Given input audio, manipulate a model feature on the fly and return both the original whisper output and output 
    after substuting in the manipulated feature.

    :param audio: Audio array
    :param whisper_model: Whisper model
    :param sae_model: SAE model
    :param whisper_subbed: Whisper model with a substituted activation
    :param feat_idx: Index of the feature to manipulate
    :param manipulation_factor: Factor to manipulate the feature by
    :return: Tuple of
    - Original whisper output (if sae_model is not None)
    - Substituted whisper output for manipulated feature
    - Substituted whisper output without manipulation
    - Substituted activation tensor without manipulation
    - Substituted activation tensor with manipulation
    :requires: audio_array must be sampled at 16kHz
    """
    mel = get_mels_from_np_array(whisper_cache.device, audio_array)
    baseline_result = whisper_cache.forward(mel)
    activations = whisper_cache.activations.to(whisper_cache.device)
    if sae_model:
        output = sae_model.forward(activations)
        if isinstance(output.encoded, L1EncoderOutput):
            manipulated_value = output.encoded.latent[:,
                                                      :, feat_idx] * manipulation_factor
            manipulated_encoding = output.encoded.latent.clone()
            manipulated_encoding[:, :, feat_idx] = manipulated_value
            manipulated_decoded = sae_model.decode(manipulated_encoding)
            standard_decoded = sae_model.decode(output.encoded.latent)
        else:
            top_acts = output.encoded.top_acts.squeeze()
            top_indices = output.encoded.top_indices.squeeze()
            manipulated_top_acts = top_acts.clone()
            for i, (idx_at_t, act_at_t) in enumerate(zip(top_indices, top_acts)):
                if feat_idx in idx_at_t:
                    idx = (idx_at_t == feat_idx).nonzero().item()
                    manipulated_value = act_at_t[idx] * manipulation_factor
                    manipulated_top_acts[i, idx] = manipulated_value
            manipulated_decoded = sae_model.decode(
                manipulated_top_acts.unsqueeze(0), top_indices.unsqueeze(0))
            standard_decoded = sae_model.decode(
                top_acts.unsqueeze(0), top_indices.unsqueeze(0))
    else:
        manipulated_value = activations[:, :, feat_idx] * manipulation_factor
        manipulated_encoding = activations.clone()
        manipulated_encoding[:, :, feat_idx] = manipulated_value
        manipulated_decoded = manipulated_encoding
        standard_decoded = activations
    manipulated_subbed_result = whisper_subbed.forward(
        mel, manipulated_decoded)
    standard_subbed_result = whisper_subbed.forward(mel, standard_decoded)
    baseline_text = None if sae_model is None else baseline_result.text
    activations_at_index = activations[:, :, feat_idx].squeeze()
    manipulated_decoded_at_index = manipulated_decoded[:, :, feat_idx].squeeze(
    )
    activation_trim_length = activation_length_from_audio_array(audio_array)
    activations_at_index_trimmed = activations_at_index[:activation_trim_length].cpu(
    )
    manipulated_decoded_at_index_trimmed = manipulated_decoded_at_index[:activation_trim_length].cpu(
    )
    return baseline_text, manipulated_subbed_result.text, standard_subbed_result.text, activations_at_index_trimmed, \
        manipulated_decoded_at_index_trimmed

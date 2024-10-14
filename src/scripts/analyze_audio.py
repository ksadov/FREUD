import torch
from typing import Optional
import argparse
from tqdm import tqdm

from src.utils.audio_utils import get_mels_from_audio_path
from src.models.hooked_model import WhisperActivationCache, init_cache
from src.models.l1autoencoder import L1AutoEncoder, L1EncoderOutput
from src.models.topkautoencoder import TopKAutoEncoder
from src.utils.activations import activation_tensor_from_indexed
from src.dataset.activations import init_sae_from_checkpoint

def analyze_audio(audio_path: str, whisper_cache: WhisperActivationCache,
                  sae_model: Optional[L1AutoEncoder | TopKAutoEncoder], 
                  top_n: int, device: str) -> tuple[list[int], list[float]]:
    """
    Given input audio, get the top features encoded by the whisper model and optionally the SAE model.
    
    :param audio: Audio file path
    :param whisper_model: Whisper model
    :param sae_model: SAE model
    :param top_n: Number of top features to return
    :return: Tuple of top feature indices and their corresponding values
    """

    mel = get_mels_from_audio_path(device, audio_path)
    whisper_cache.forward(mel)
    activations = whisper_cache.activations
    indexed_activations = False
    if sae_model:
        output = sae_model.forward(activations)
        if isinstance(output.encoded, L1EncoderOutput):
            activations = output.encoded.latent
        else:
            top_acts = output.encoded.top_acts
            top_indices = output.encoded.top_indices
            indexed_activations = True
    if indexed_activations:
        raise("TODO: Implement indexed activations")
    else:
        activations = activations.squeeze()
        unique_top_activations = []
        for activations_at_timestep in activations:
            top_n_at_t = activations_at_timestep.topk(top_n)
            unique_top_activations.extend([(i.item(), v.item()) for i, v in zip(top_n_at_t.indices, top_n_at_t.values)])
            # sort by value
            unique_top_activations = sorted(unique_top_activations, key=lambda x: x[1], reverse=True)
            new_unique = []
            for i, (idx, value) in enumerate(unique_top_activations):
                if idx not in [idx for idx, _ in new_unique] and len(new_unique) < top_n:
                    new_unique.append((idx, value))
            unique_top_activations = new_unique

        # sanity check
        max_activations = []
        for i, v in unique_top_activations:
            max_at_i = activations[:, i].max()
            assert max_at_i == v, f"Max activation at index {i} is {max_at_i}, expected {v}"
            max_activations.append(activations[:, i])
        return [i for i, _ in unique_top_activations], max_activations

def init_analysis_models(whisper_model: str, sae_path: str, layer_to_cache: str, device: torch.device) -> \
  tuple[WhisperActivationCache, Optional[L1AutoEncoder | TopKAutoEncoder]]:
    """
    Initialize the Whisper model and optionally the SAE model for analysis
    
    :param whisper_model: Whisper model name
    :param sae_path: SAE model path
    :param layer_to_cache: Layer to cache
    :param device: Device to use
    :return: Tuple of Whisper cache and SAE model
    """
    whisper_cache = init_cache(whisper_model, layer_to_cache, device)
    sae_model = init_sae_from_checkpoint(sae_path) if sae_path else None
    if sae_model:
        sae_model.eval()
        sae_model.to(device)
    return whisper_cache, sae_model

def main():
    parser = argparse.ArgumentParser(description="Analyze audio using Whisper and optionally SAE")
    parser.add_argument("--audio_path", type=str, help="Path to audio file", default="audio_data/librispeech/LibriSpeech/test-other/1688/142285/1688-142285-0000.flac")
    parser.add_argument("--whisper_model", type=str, help="name of whisper model", default="tiny")
    parser.add_argument("--layer_to_cache", type=str, help="Layer to cache", default="encoder.blocks.2.mlp.1")
    parser.add_argument("--sae_path", type=str, default=None, help="Path to SAE model")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top features to return")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    args = parser.parse_args()

    whisper_cache, sae_model = init_analysis_models(args.whisper_model, args.sae_path, args.layer_to_cache, args.device)
    top = analyze_audio(args.audio_path, whisper_cache, sae_model, args.top_n, args.device)
    print(f"Top features: {top}")

if __name__ == "__main__":
    main()
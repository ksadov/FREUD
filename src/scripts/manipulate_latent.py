import torch
from typing import Optional
import argparse
from tqdm import tqdm
import numpy as np
import soundfile as sf

from src.utils.audio_utils import get_mels_from_np_array
from src.models.hooked_model import WhisperActivationCache, init_cache, WhisperSubbedActivation, init_subbed
from src.models.l1autoencoder import L1AutoEncoder, L1EncoderOutput
from src.models.topkautoencoder import TopKAutoEncoder
from src.dataset.activations import init_sae_from_checkpoint
from src.utils.constants import SAMPLE_RATE

def manipulate_latent(audio_array: np.ndarray, whisper_cache: WhisperActivationCache,
                  sae_model: Optional[L1AutoEncoder | TopKAutoEncoder], 
                  whisper_subbed: WhisperSubbedActivation, feat_idx: int, 
                  manipulation_factor: float) -> tuple[str, str, str, torch.Tensor, torch.Tensor]:
    """
    Given input audio, manipulate a model feature on the fly and return both the original whisper output and output 
    after substuting in the manipulated feature.

    :param audio: Audio file path
    :param whisper_model: Whisper model
    :param sae_model: SAE model
    :param whisper_subbed: Whisper model with a substituted activation
    :param feat_idx: Index of the feature to manipulate
    :param manipulation_factor: Factor to manipulate the feature by
    :return: Tuple of
    - Original whisper output
    - Substituted whisper output
    - Substituted whisper output with standard decoding
    - Original activation tensor
    - Substituted activation tensor
    """
    mel = get_mels_from_np_array(whisper_cache.device, audio_array)
    baseline_result = whisper_cache.forward(mel)
    activations = whisper_cache.activations
    if sae_model:
        output = sae_model.forward(activations)
        if isinstance(output.encoded, L1EncoderOutput):
            manipulated_value = output.encoded.latent[:, :, feat_idx] * manipulation_factor
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
            manipulated_decoded = sae_model.decode(manipulated_top_acts.unsqueeze(0), top_indices.unsqueeze(0))
            standard_decoded = sae_model.decode(top_acts.unsqueeze(0), top_indices.unsqueeze(0))
    else:
        manipulated_value = activations[:, :, feat_idx] * manipulation_factor
        manipulated_encoding = activations.clone()
        manipulated_encoding[:, :, feat_idx] = manipulated_value
        manipulated_decoded = manipulated_encoding
        standard_decoded = activations
    manipulated_subbed_result = whisper_subbed.forward(mel, manipulated_decoded)
    standard_subbed_result = whisper_subbed.forward(mel, standard_decoded)
    return baseline_result.text, manipulated_subbed_result.text, standard_subbed_result.text, activations, manipulated_decoded


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
    sae_model = init_sae_from_checkpoint(sae_path, device) if sae_path else None
    if sae_model:
        sae_model.eval()
    return whisper_cache, sae_model

def main():
    parser = argparse.ArgumentParser(description="Analyze audio using Whisper and optionally SAE")
    parser.add_argument("--audio_path", type=str, help="Path to audio file", default="audio_data/librispeech/LibriSpeech/test-other/1688/142285/1688-142285-0000.flac")
    parser.add_argument("--whisper_model", type=str, help="name of whisper model", default="tiny")
    parser.add_argument("--layer_to_cache", type=str, help="Layer to cache", default="encoder.blocks.2")
    parser.add_argument("--sae_path", type=str, default="runs/topkautoencoder_baseline/checkpoints/bestval.pth", help="Path to SAE model")
    parser.add_argument("--device", type=str, help="Device to run on", default="cuda:0")
    parser.add_argument("--feat_idx", type=int, help="Feature index to manipulate", default=0)
    parser.add_argument("--manipulation_factor", type=float, help="Factor to manipulate the feature by", default=1.5)
    args = parser.parse_args()

    audio_array, sr = sf.read(args.audio_path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate of {SAMPLE_RATE} but got {sr}")
    
    whisper_cache, sae_model = init_analysis_models(args.whisper_model, args.sae_path, args.layer_to_cache, args.device)
    whisper_subbed = init_subbed(args.whisper_model, args.layer_to_cache, args.device)
    before, after, standard, vec_before, vec_after = manipulate_latent(audio_array, whisper_cache, sae_model, 
                                                                       whisper_subbed, args.feat_idx, 
                                                                       args.manipulation_factor)
    print(f"Before: {before}")
    print(f"After: {after}")
    print(f"Standard: {standard}")
    print(f"Before: {vec_before}")
    print(f"After: {vec_after}")

    

if __name__ == "__main__":
    main()
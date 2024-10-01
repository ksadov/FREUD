import argparse
import torch
import whisper
from librispeech_data import get_mels_from_audio_path
from hooked_model import WhisperSubbedActivation, WhisperActivationCache

def init_models(whisper_model: str, layer_name: str, device: torch.device):
  whisper_model = whisper.load_model(whisper_model)
  whisper_cache = WhisperActivationCache(
        model=whisper_model,
        activation_regex=[layer_name + "$"],
        device=device
    )
  whisper_sub = WhisperSubbedActivation(
    model=whisper_model,
    substitution_layer=layer_name,
    device=device
  )
  return whisper_cache, whisper_sub
 
def test_cached(
        layer_name: str,
        whisper_cache: WhisperActivationCache, 
        whisper_sub: WhisperSubbedActivation, 
        file_name: str, 
        device: torch.device
        ):
  mels = get_mels_from_audio_path(device, file_name).unsqueeze(0)
  with torch.no_grad():
          result = whisper_cache.forward(mels)
  activation = whisper_cache.activations[layer_name]
  print("result: ", result)
  subbed_result = whisper_sub.forward(mels, activation)
  print("subbed result: ", subbed_result)
  redo_result = whisper_sub.forward(mels, None)
  print("redo result: ", redo_result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name")
    parser.add_argument("--whisper_model", default="tiny")
    parser.add_argument("--layer_name", default="encoder.blocks.2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    whisper_cache, whisper_sub = init_models(args.whisper_model, args.layer_name, device)
    test_cached(args.layer_name, whisper_cache, whisper_sub, args.file_name, device)

if __name__ == "__main__":
    main()

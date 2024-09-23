import pathlib
import argparse
import torch
from hooked_model import WhisperActivationCache
from librispeech_data import LibriSpeechDataset
from torch.utils.data import DataLoader
import whisper
import json
from tqdm import tqdm
import multiprocessing
from datetime import datetime


def create_out_folder(folder_name):
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    return folder_name


def save_batch(batch, out_folder, batch_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(batch, f"{out_folder}/batch_{batch_id}_{timestamp}.pt")


def get_activations(
    data_path: str,
    whisper_model: torch.nn.Module,
    layers_to_cache: list[str],
    split: str,
    batch_size: int,
    device: torch.device,
    out_folder_prefix: str
):
    whisper_cache = WhisperActivationCache(
        model=whisper_model,
        activation_regex=layers_to_cache,
        device=device,
    )
    whisper_cache.model.eval()
    dataset = LibriSpeechDataset(data_path, split, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    batch_id = 0
    for batch in tqdm(dataloader):
        whisper_cache.reset_state()
        mels, utterance_ids, _, transcript = batch
        with torch.no_grad():
            result = whisper_cache.forward(mels)
        activations = whisper_cache.activations
        print("activations", activations.keys())
        for name, act in activations.items():
            print("Saving activations for", name)
            out_folder = create_out_folder(
                f"{out_folder_prefix}/{split}/{name}")
            activation_batch = {}
            for i, utterance_id in enumerate(utterance_ids):
                print("transcript for", utterance_id, transcript[i])
                print("result for ", utterance_id, result[i].text)
                activation_batch[utterance_id] = (act[i], result[i].text)
            if activation_batch:
                save_batch((activation_batch),
                           out_folder, batch_id)
        batch_id += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="tiny.json")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    whisper_model = whisper.load_model(config["whisper_model"])
    print("Named modules", [name for name, _ in whisper_model.named_modules()])
    for split in config["splits"]:
        print(f"Processing split {split}")
        get_activations(
            config["data_path"],
            whisper_model,
            config["layers"],
            split,
            config["batch_size"],
            torch.device(config["device"]),
            config["out_folder_prefix"],
        )


if __name__ == "__main__":
    main()

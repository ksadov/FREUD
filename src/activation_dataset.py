import torch
import os
from feature_api import load_activations
from librispeech_data import LibriSpeechDataset
from hooked_model import WhisperActivationCache
from torch.utils.data import DataLoader


class ActivationDataset(torch.utils.data.Dataset):

    def __init__(self, activation_folder: str, split: str):
        super().__init__()
        self.activation_folder = activation_folder
        activation_dict = load_activations(activation_folder)
        self.activation_list = list(activation_dict.items())
        self.split = split
        self.activation_shape = self._get_activation_shape()

    def _get_activation_shape(self):
        audio_fname, activation = self.activation_list[0]
        return activation.shape

    def __getitem__(self, idx) -> torch.Tensor:
        audio_fname, activation = self.activation_list[idx]
        return activation, audio_fname

    def __len__(self) -> int:
        return len(self.activation_list)


class FlyActivationDataset(torch.utils.data.Dataset):
    def __init__(self,  data_path: str, whisper_model: torch.nn.Module, layer_to_cache: str, device: torch.device, split: str):
        self.whisper_cache = WhisperActivationCache(
            model=whisper_model,
            activation_regex=[layer_to_cache + "$"],
            device=device,
        )
        self.whisper_cache.model.eval()
        self.dataset = LibriSpeechDataset(data_path, split, device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.activation_shape = self._get_activation_shape()

    def _get_activation_shape(self):
        mels, _, _, _ = self.dataset[0]
        with torch.no_grad():
            result = self.whisper_cache.forward(mels)
        activations = self.whisper_cache.activations
        for name, act in activations.items():
            return act.shape

    def __getitem__(self, idx) -> torch.Tensor:
        mels, _, global_file_name, transcript = self.dataset[idx]
        with torch.no_grad():
            result = self.whisper_cache.forward(mels)
        activations = self.whisper_cache.activations
        for name, act in activations.items():
            return act, global_file_name

    def __len__(self) -> int:
        return len(self.dataset)


class FlyActivationDataloader(torch.utils.data.DataLoader):
    def __init__(self,  data_path: str, whisper_model: torch.nn.Module, layer_to_cache: str, device: torch.device, split: str, dl_kwargs: dict):
        self.whisper_cache = WhisperActivationCache(
            model=whisper_model,
            activation_regex=[layer_to_cache + "$"],
            device=device,
        )
        self.whisper_cache.model.eval()
        self.ls_dataset = LibriSpeechDataset(data_path, split, device)
        self.ls_dataloader = DataLoader(
            self.ls_dataset, **dl_kwargs)
        self.activation_shape = self._get_activation_shape()

    def _get_activation_shape(self):
        mels, _, _, _ = self.ls_dataset[0]
        with torch.no_grad():
            result = self.whisper_cache.forward(mels)
        activations = self.whisper_cache.activations
        for name, act in activations.items():
            print("SHAPE", act.shape)
            return act.squeeze().shape

    def __iter__(self):
        for batch in self.ls_dataloader:
            mels, _, global_file_name, transcript = batch
            with torch.no_grad():
                result = self.whisper_cache.forward(mels)
            activations = self.whisper_cache.activations
            for name, act in activations.items():
                yield act, global_file_name

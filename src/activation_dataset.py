import torch
import os
from feature_api import load_activations

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
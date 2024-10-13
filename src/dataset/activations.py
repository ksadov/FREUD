import torch
import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from src.dataset.audio import AudioDataset
from src.models.hooked_model import init_cache
from src.models.l1autoencoder import L1AutoEncoder
from src.models.topkautoencoder import TopKAutoEncoder
from src.models.config import L1AutoEncoderConfig, TopKAutoEncoderConfig


def init_sae_from_checkpoint(checkpoint: str) -> L1AutoEncoder | TopKAutoEncoder:
    checkpoint = torch.load(checkpoint)
    activation_size = checkpoint['hparams']['activation_size']
    if checkpoint['hparams']['autoencoder_variant'] == 'l1':
        cfg = L1AutoEncoderConfig.from_dict(
            checkpoint['hparams']['autoencoder_config'])
        model = L1AutoEncoder(activation_size, cfg)
    else:
        cfg = TopKAutoEncoderConfig.from_dict(
            checkpoint['hparams']['autoencoder_config'])
        model = TopKAutoEncoder(activation_size, cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


class FlyActivationDataLoader(torch.utils.data.DataLoader):
    """
    Dataloader for computing Whisper or SAE activations on the fly
    """

    def __init__(self,  data_path: str, whisper_model: str, sae_checkpoint: Optional[str],
                 layer_name: str, device: str, batch_size: int, dl_max_workers: int,
                 subset_size: Optional[int] = None):
        self.whisper_cache = init_cache(whisper_model, layer_name, device)
        self.whisper_cache.model.eval()
        self.sae_model = init_sae_from_checkpoint(
            sae_checkpoint) if sae_checkpoint else None
        if isinstance(self.sae_model, TopKAutoEncoder):
            self.activation_type = "indexed"
        else:
            self.activation_type = "tensor"
        self._dataset = AudioDataset(data_path, device)
        if subset_size:
            self._dataset = torch.utils.data.Subset(
                self._dataset, range(subset_size))
        dl_kwargs = {
            "batch_size": batch_size,
            "pin_memory": False,
            "drop_last": True,
            "num_workers": dl_max_workers,
        }
        self._dataloader = DataLoader(self._dataset, **dl_kwargs)
        self.activation_shape = self._get_activation_shape()
        self.dataset_length = len(self._dataset)

    def _get_activation_shape(self):
        mels, _ = self._dataset[0]
        with torch.no_grad():
            self.whisper_cache.forward(mels)
            first_activation = self.whisper_cache.activations[0]
            if isinstance(self.sae_model, L1AutoEncoder):
                encoded = self.sae_model.encode(first_activation)
                return encoded.latent.squeeze().shape
            elif isinstance(self.sae_model, TopKAutoEncoder):
                encoded = self.sae_model.encode(first_activation)
                return encoded.top_acts.squeeze().shape
            else:
                return first_activation.squeeze().shape

    def __iter__(self):
        for batch in self._dataloader:
            self.whisper_cache.reset_state()
            mels, global_file_names = batch
            self.whisper_cache.forward(mels)
            activations = self.whisper_cache.activations
            if isinstance(self.sae_model, L1AutoEncoder):
                encoded = self.sae_model.encode(activations)
                yield encoded.latent, global_file_names
            elif isinstance(self.sae_model, TopKAutoEncoder):
                encoded = self.sae_model.encode(activations)
                yield encoded.top_acts, encoded.top_indices, global_file_names
            else:
                yield activations, global_file_names

    def __len__(self):
        return len(self._dataloader)


class MemoryMappedActivationsDataset(Dataset):
    """
    Dataset for activations stored in memory-mapped files geneerated by src.scripts.collect_activations
    """

    def __init__(self, data_path: str, layer_name: str, subset_size: Optional[int] = None):
        self.data_path = data_path
        self.layer_name = layer_name
        self.metadata_file = os.path.join(
            data_path, f"{layer_name}_metadata.json")
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        tensor_path = os.path.join(data_path, f"{layer_name}_tensors.npy")
        if not os.path.exists(tensor_path):
            activation_value_file = os.path.join(
                data_path, f"{layer_name}_activation_values.npy")
            feature_index_file = os.path.join(
                data_path, f"{layer_name}_feature_indices.npy")
            self.activation_type = "indexed"
            self.act_mmap = np.load(activation_value_file, mmap_mode='r')
            self.idx_mmap = np.load(feature_index_file, mmap_mode='r')
        else:
            self.activation_type = "tensor"
            self.mmap = np.load(self.tensor_file, mmap_mode='r')
        if subset_size is not None:
            self.metadata['filenames'] = self.metadata['filenames'][:subset_size]
            self.metadata['tensor_shapes'] = self.metadata['tensor_shapes'][:subset_size]
            self.mmap = self.mmap[:subset_size]
        self.activation_shape = self._get_activation_shape()

    def _get_activation_shape(self):
        return self.metadata['tensor_shapes'][0]

    def __len__(self):
        return len(self.metadata['filenames'])

    def __getitem__(self, idx):
        filename = self.metadata['filenames'][idx]
        tensor_shape = self.metadata['tensor_shapes'][idx]

        if self.activation_type == "indexed":
            act_data = self.act_mmap[idx]
            act_data = torch.from_numpy(act_data.reshape(tensor_shape))
            idx_data = self.idx_mmap[idx]
            idx_data = torch.from_numpy(idx_data.reshape(tensor_shape))
            return act_data, idx_data, filename
        else:
            tensor_data = self.mmap[idx]
            tensor = torch.from_numpy(tensor_data.reshape(tensor_shape))

            return tensor, filename


class MemoryMappedActivationDataLoader(torch.utils.data.DataLoader):
    """
    Dataloader for activations stored in memory-mapped files generated by src.scripts.collect_activations
    """

    def __init__(self, data_path: str, layer_name: str, batch_size: int, dl_max_workers: int, subset_size: Optional[int] = None):
        self._dataset = MemoryMappedActivationsDataset(
            data_path, layer_name, subset_size)
        dl_kwargs = {
            "batch_size": batch_size,
            "pin_memory": False,
            "drop_last": True,
            "num_workers": dl_max_workers,
        }
        super().__init__(self._dataset, **dl_kwargs)
        self.activation_shape = self.dataset.activation_shape
        self.activation_type = self.dataset.activation_type
        self.dataset_length = len(self._dataset)

    def __len__(self):
        return len(self._dataset) // self.batch_size

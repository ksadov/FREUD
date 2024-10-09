import torch
from librispeech_data import LibriSpeechDataset
from hooked_model import init_cache
from torch.utils.data import DataLoader
from typing import Optional
from autoencoder import init_from_checkpoint

class FlyActivationDataloader(torch.utils.data.DataLoader):
    def __init__(self,  data_path: str, whisper_model: torch.nn.Module, sae_checkpoint: Optional[str], 
                 layer_to_cache: str, device: torch.device, split: str, batch_size: int, dl_max_workers: int,
                 subset_size: Optional[int] = None):
        self.whisper_cache = init_cache(whisper_model, [layer_to_cache], device)
        self.whisper_cache.model.eval()
        self.sae_model = init_from_checkpoint(sae_checkpoint) if sae_checkpoint else None
        self.ls_dataset = LibriSpeechDataset(data_path, split, device)
        if subset_size:
            self.ls_dataset = torch.utils.data.Subset(self.ls_dataset, range(subset_size))
        dl_kwargs = {
            "batch_size": batch_size,
            "pin_memory": False,
            "drop_last": True,
            "num_workers": dl_max_workers,
        }
        self.ls_dataloader = DataLoader(
            self.ls_dataset, **dl_kwargs)
        self.activation_shape = self._get_activation_shape()

    def _get_activation_shape(self):
        mels, _, _, _ = self.ls_dataset[0]
        with torch.no_grad():
            self.whisper_cache.forward(mels)
            activations = self.whisper_cache.activations
            _, first_activation = next(iter(activations.items()))
            if self.sae_model:
                _, c = self.sae_model(first_activation)
                return c.squeeze().shape
            else:
                return first_activation.squeeze().shape

    def __iter__(self):
        for batch in self.ls_dataloader:
            self.whisper_cache.reset_state()
            mels, _, global_file_name, transcript = batch
            self.whisper_cache.forward(mels)
            activations = self.whisper_cache.activations
            for name, act in activations.items():
                if self.sae_model:
                    _, c = self.sae_model(act)
                    yield global_file_name, c
                else:
                    yield global_file_name, act
    
    def __len__(self):
        return len(self.ls_dataloader)
        

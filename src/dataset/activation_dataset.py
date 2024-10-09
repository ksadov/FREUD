import torch
from torch.utils.data import DataLoader
from typing import Optional
from src.dataset.audio_dataset import AudioDataset
from src.models.hooked_model import init_cache
from src.models.autoencoder import init_from_checkpoint

class FlyActivationDataloader(torch.utils.data.DataLoader):
    def __init__(self,  data_path: str, whisper_model: torch.nn.Module, sae_checkpoint: Optional[str], 
                 layer_to_cache: str, device: torch.device, batch_size: int, dl_max_workers: int,
                 subset_size: Optional[int] = None):
        self.whisper_cache = init_cache(whisper_model, layer_to_cache, device)
        self.whisper_cache.model.eval()
        self.sae_model = init_from_checkpoint(sae_checkpoint) if sae_checkpoint else None
        self.dataset = AudioDataset(data_path, device)
        if subset_size:
            self.dataset = torch.utils.data.Subset(self.dataset, range(subset_size))
        dl_kwargs = {
            "batch_size": batch_size,
            "pin_memory": False,
            "drop_last": True,
            "num_workers": dl_max_workers,
        }
        self.dataloader = DataLoader(self.dataset, **dl_kwargs)
        self.activation_shape = self._get_activation_shape()

    def _get_activation_shape(self):
        mels, _ = self.dataset[0]
        with torch.no_grad():
            self.whisper_cache.forward(mels)
            first_activation = self.whisper_cache.activations[0]
            if self.sae_model:
                _, c = self.sae_model(first_activation)
                return c.squeeze().shape
            else:
                return first_activation.squeeze().shape

    def __iter__(self):
        for batch in self.dataloader:
            self.whisper_cache.reset_state()
            mels, global_file_names = batch
            self.whisper_cache.forward(mels)
            activations = self.whisper_cache.activations
            if self.sae_model:
                _, c = self.sae_model(activations)
                yield c, global_file_names
            else:
                yield activations, global_file_names
    
    def __len__(self):
        return len(self.dataloader)
        

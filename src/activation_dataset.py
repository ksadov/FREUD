import torch

# for now, we'll just return random vectors for the activations
class ActivationDataset(torch.utils.data.Dataset):

    def __init__(self, activation_folder: str, split: str):
        super().__init__()
        self.activation_folder = activation_folder
        self.activation_shape = (128, 256)
        self.n_samples = 1000
        self.split = split

    def __getitem__(self, idx) -> torch.Tensor:
        fname = f"activation_{idx}.wav"
        activation = torch.randn(self.activation_shape)
        return activation, fname

    def __len__(self) -> int:
        return self.n_samples
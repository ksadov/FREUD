import torch
from jaxtyping import Float
from torch import Tensor, nn
from src.models.hooked_model import WhisperActivationCache, activations_from_audio


class AutoEncoder(nn.Module):
    def __init__(self, hp: dict):
        """
        Autoencoder model for audio features

        requires: hp must contain the following
            - activation_size: int
            - n_dict_components: int
        """
        super(AutoEncoder, self).__init__()
        self.hp = hp
        self.tied = True  # tie encoder and decoder weights
        self.activation_size = hp['activation_size']
        self.n_dict_components = hp['n_dict_components']

        # Only defining the decoder layer, encoder will share its weights
        self.decoder = nn.Linear(self.n_dict_components, self.activation_size, bias=False)
        # Create a bias layer
        self.encoder_bias = nn.Parameter(torch.zeros(self.n_dict_components))

        # Initialize the decoder weights orthogonally
        nn.init.orthogonal_(self.decoder.weight)

        # Encoder is a Sequential with the ReLU activation
        # No need to define a Linear layer for the encoder as its weights are tied with the decoder
        self.encoder = nn.Sequential(nn.ReLU())

    def forward(self, x: Float[Tensor, "bsz seq_len d_model"]):  # noqa: F821
        # Apply unit norm constraint to the decoder weights
        self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)
        c = self.encoder(x @ self.decoder.weight + self.encoder_bias)

        # Decoding step as before
        x_hat = self.decoder(c)
        return x_hat, c
    
def init_from_checkpoint(checkpoint: str):
    checkpoint = torch.load(checkpoint)
    hp = checkpoint['hparams']
    model = AutoEncoder(hp)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def get_audio_features(sae_model: AutoEncoder, whisper_cache: WhisperActivationCache, audio_fname: str):
    activations, _ = activations_from_audio(whisper_cache, audio_fname)
    activation_values = torch.cat(list(activations.values()), dim=1)
    _, c = sae_model(activation_values)
    return c
    
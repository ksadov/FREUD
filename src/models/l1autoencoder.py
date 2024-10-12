from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.models.hooked_model import WhisperActivationCache, activations_from_audio

# modified from
# https://github.com/er537/whisper_interpretability/tree/master/whisper_interpretability/sparse_coding/train/autoencoder.py


class L1EncoderOutput(NamedTuple):
    latent: Tensor


class L1ForwardOutput(NamedTuple):
    sae_out: Tensor

    encoded: L1EncoderOutput

    l1_loss: Tensor

    reconstruction_loss: Tensor


def mse_loss(input, target, ignored_index, reduction):
    # mse_loss with ignored_index
    mask = target == ignored_index
    out = (input[~mask] - target[~mask]) ** 2
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out


class L1AutoEncoder(nn.Module):
    def __init__(self, hp: dict):
        """
        Autoencoder model for audio features

        :param hp: dictionary containing hyperparameters
        :requires: hp must contain the following keys:
            - activation_size: size of the activation layer
            - n_dict_components: number of dictionary components
        """
        super(L1AutoEncoder, self).__init__()
        self.hp = hp
        self.tied = True  # tie encoder and decoder weights
        self.activation_size = hp['activation_size']
        self.n_dict_components = hp['n_dict_components']
        self.recon_alpha = hp['recon_alpha']

        # Only defining the decoder layer, encoder will share its weights
        self.decoder = nn.Linear(
            self.n_dict_components, self.activation_size, bias=False)
        # Create a bias layer
        self.encoder_bias = nn.Parameter(torch.zeros(self.n_dict_components))

        # Initialize the decoder weights orthogonally
        nn.init.orthogonal_(self.decoder.weight)

        # Encoder is a Sequential with the ReLU activation
        # No need to define a Linear layer for the encoder as its weights are tied with the decoder
        self.encoder = nn.Sequential(nn.ReLU())

    def encode(self, x: Float[Tensor, "bsz seq_len d_model"]):  # noqa: F821
        # Apply unit norm constraint to the decoder weights
        self.decoder.weight.data = nn.functional.normalize(
            self.decoder.weight.data, dim=0)
        c = self.encoder(x @ self.decoder.weight + self.encoder_bias)
        return L1EncoderOutput(latent=c)

    def forward(self, x: Float[Tensor, "bsz seq_len d_model"]):  # noqa: F821
        c = self.encode(x).latent
        x_hat = self.decoder(c)
        loss_l1 = torch.norm(c, 1, dim=2).mean()
        loss_recon = self.recon_alpha * mse_loss(x_hat, x, -1, "mean")
        return L1ForwardOutput(sae_out=x_hat, encoded=c, l1_loss=loss_l1, reconstruction_loss=loss_recon)

    @staticmethod
    def init_from_checkpoint(checkpoint: str):
        checkpoint = torch.load(checkpoint)
        hp = checkpoint['hparams']
        model = L1AutoEncoder(hp)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model


def get_audio_features(sae_model: L1AutoEncoder, whisper_cache: WhisperActivationCache, audio_fname: str):
    activations, _ = activations_from_audio(whisper_cache, audio_fname)
    activation_values = torch.cat(list(activations.values()), dim=1)
    out = sae_model.encode(activation_values)
    return out.latent

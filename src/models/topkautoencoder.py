from typing import NamedTuple

import einops
import torch
from torch import Tensor, nn

from src.models.config import TopKAutoEncoderConfig
from src.utils.models import get_n_dict_components


# modified from https://github.com/EleutherAI/sae/tree/main/sae/sae.py


# via https://github.com/EleutherAI/sae/tree/main/sae/utils.py
def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts @ W_dec.mT


class TopKEncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


class TopKForwardOutput(NamedTuple):
    sae_out: Tensor

    encoded: TopKEncoderOutput

    fvu: Tensor
    """Fraction of variance unexplained."""

    auxk_loss: Tensor
    """AuxK loss, if applicable."""

    multi_topk_fvu: Tensor
    """Multi-TopK FVU, if applicable."""


class TopKAutoEncoder(nn.Module):
    def __init__(
        self, activation_size: int, cfg: TopKAutoEncoderConfig, decoder: bool = True
    ):
        """
        Autoencoder model for audio features

        :param hp: dictionary containing hyperparameters
        :requires: hp must contain the following keys:
            - activation_size: size of the activation layer
            - n_dict_components: number of dictionary components
        """
        super().__init__()
        self.cfg = cfg
        self.d_in = activation_size
        self.n_dict_components = self.n_dict_components = get_n_dict_components(
            activation_size, cfg.expansion_factor, cfg.n_dict_components
        )

        self.encoder = nn.Linear(self.d_in, self.n_dict_components)
        self.encoder.bias.data.zero_()

        self.W_dec = nn.Parameter(self.encoder.weight.data.clone()) if decoder else None
        if decoder and self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(self.d_in))

    def pre_acts(self, x: Tensor) -> Tensor:
        # Remove decoder bias as per Anthropic
        sae_in = x - self.b_dec
        out = self.encoder(sae_in)

        return nn.functional.relu(out)

    def select_topk(self, latents: Tensor) -> TopKEncoderOutput:
        """Select the top-k latents."""
        return TopKEncoderOutput(*latents.topk(self.cfg.k, sorted=False))

    def encode(self, x: Tensor) -> TopKEncoderOutput:
        """Encode the input and select the top-k latents."""
        return self.select_topk(self.pre_acts(x))

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        y = eager_decode(top_indices, top_acts, self.W_dec.mT)
        return y + self.b_dec

    def forward(
        self, x: Tensor, dead_mask: Tensor | None = None, return_mse: bool = False
    ) -> TopKForwardOutput | tuple[TopKForwardOutput, Tensor]:
        pre_acts = self.pre_acts(x)

        # Decode and compute residual
        top_acts, top_indices = self.select_topk(pre_acts)
        sae_out = self.decode(top_acts, top_indices)
        e = sae_out - x

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (x - x.mean(0)).pow(2).sum()
        if total_variance == 0:
            total_variance = 1.0

        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = x.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        if self.cfg.multi_topk:
            top_acts, top_indices = pre_acts.topk(4 * self.cfg.k, sorted=False)
            sae_out = self.decode(top_acts, top_indices)

            multi_topk_fvu = (sae_out - x).pow(2).sum() / total_variance
        else:
            multi_topk_fvu = sae_out.new_tensor(0.0)

        forward_output = TopKForwardOutput(
            sae_out,
            TopKEncoderOutput(top_acts, top_indices),
            fvu,
            auxk_loss * self.cfg.auxk_alpha,
            multi_topk_fvu,
        )
        if return_mse:
            return forward_output, e.pow(2).mean()
        return forward_output

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

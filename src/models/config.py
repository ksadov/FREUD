from dataclasses import dataclass
from simple_parsing import Serializable


@dataclass
class AutoEncoderConfig(Serializable):
    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""
    n_dict_components: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""


@dataclass
class L1AutoEncoderConfig(AutoEncoderConfig):
    recon_alpha: float = 1.0
    """Weight of the reconstruction loss."""


@dataclass
class TopKAutoEncoderConfig(AutoEncoderConfig):
    normalize_decoder: bool = True
    """Whether to normalize the decoder weights to unit norm."""
    k: int = 32
    """Number of top latents to keep."""
    multi_topk: bool = False
    """Whether to use multi-topk."""
    auxk_alpha: float = 0.0
    """Weight of the auxk loss."""

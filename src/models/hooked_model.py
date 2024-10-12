from abc import ABC, abstractmethod
from datetime import date
from typing import Callable, Optional
import torch
from jaxtyping import Float
from torch import Tensor
import whisper

from src.utils.audio_utils import get_mels_from_audio_path


class BaseActivationModule(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        layer_to_cache: str,
        hook_fn: Optional[Callable] = None,
    ):
        """
        Base class using pytorch hooks to cache intermediate activations at a specified layer
        Parent classes should inherit from this class, implementing their own custom_forward method
        You can optionally pass in your own hook_fn
        """
        assert model is not None, "no model found"
        self.model = model
        self.step = 0
        self.activations = None
        self.hooks = []
        self.layer_to_cache = layer_to_cache
        self.hook_fn = hook_fn

    def forward(self, x: Float[Tensor, "bsz seq_len n_mels"]):  # noqa: F821
        self.model.zero_grad()
        self.step += 1
        self.register_hooks()
        model_out = self.custom_forward(self.model, x)
        self.remove_hooks()
        return model_out

    def substituted_forward(self, x: Float[Tensor, "bsz seq_len n_mels"], substituted_activation: torch.tensor):
        pass

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.layer_to_cache:
                hook_fn = self.hook_fn if self.hook_fn is not None else self._get_caching_hook(
                    name)
                forward_hook = module.register_forward_hook(hook_fn)
                self.hooks.append(forward_hook)

    def _get_caching_hook(self, name):
        def hook(module, input, output):
            output_ = output.detach().cpu()
            # assume that we go through each layer only once
            self.activations = output_

        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @abstractmethod
    def custom_forward(
        self,
        model: torch.nn.Module,
        mels: Float[Tensor, "bsz seq_len n_mels"],  # noqa: F821
    ):
        """
        Should be overidden inside child class to match specific model.
        """
        raise NotImplementedError

    def reset_state(self):
        self.activations = None


class WhisperActivationCache(BaseActivationModule):
    """
    Use hooks in BaseActivationModule to cache intermediate activations while running forward pass
    """

    def __init__(
        self,
        layer_to_cache: str,
        hook_fn: Optional[Callable] = None,
        model: Optional[torch.nn.Module] = None,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(model, layer_to_cache, hook_fn)
        self.device = device

    def custom_forward(
        self, model: torch.nn.Module, mels: Float[Tensor, "bsz seq_len n_mels"]
    ):  # noqa: F821
        options = whisper.DecodingOptions(
            without_timestamps=False, fp16=(self.device == torch.device("cuda"))
        )
        output = model.decode(mels, options)
        return output

    def _get_caching_hook(self, name):
        # custom caching function for whisper
        def hook(module, input, output):
            output_ = output.detach().cpu()
            self.activations = output_

        return hook


class WhisperSubbedActivation(torch.nn.Module):
    """
    Whisper but we can substitute a custom activation for one of the layers
    """

    def __init__(
        self,
        model: torch.nn.Module,
        substitution_layer: str,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.substitution_layer = substitution_layer

    def forward(self, mels: Float[Tensor, "bsz seq_len n_mels"], substitute_activation: Tensor):  # noqa: F821
        self.model.zero_grad()
        if substitute_activation is not None:
            forward_hook = self.register_hook(substitute_activation)
        options = whisper.DecodingOptions(
            without_timestamps=False, fp16=(self.device == torch.device("cuda"))
        )
        output = self.model.decode(mels, options)
        if substitute_activation is not None:
            forward_hook.remove()
        return output

    def register_hook(self, substitution_activation: Tensor):
        for name, module in self.model.named_modules():
            if name == self.substitution_layer:
                hook_fn = self._get_substitution_hook(substitution_activation)
                forward_hook = module.register_forward_hook(hook_fn)
                return forward_hook

    def _get_substitution_hook(self, substitution_activation):
        def hook(module, input, output):
            sub_act = substitution_activation.to(output.dtype)
            return sub_act

        return hook


def init_cache(whisper_model: str, layer_to_cache: str, device: torch.device) -> WhisperActivationCache:
    whisper_model = whisper.load_model(whisper_model)
    whisper_model.eval()
    return WhisperActivationCache(model=whisper_model, layer_to_cache=layer_to_cache, device=device)


def activations_from_audio(model: WhisperActivationCache, audio_fname: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    with torch.no_grad():
        mel = get_mels_from_audio_path(model.device, audio_fname)
        result = model.forward(mel)
    return model.activations, result

from abc import ABC, abstractmethod
from datetime import date
from typing import Callable, Optional
import torch
from jaxtyping import Float
from torch import Tensor
import whisper
from natsort import natsorted
import regex


class BaseActivationModule(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        hook_fn: Optional[Callable] = None,
        activation_regex: list[str] = ["*.blocks.*.mlp.*"],
    ):
        """
        Base class using pytorch hooks to cache all intermediate
        activations in [activations_to_cache]
        Parent classes should inherit from this class, implementing their own custom_forward method
        You can optionally pass in your own hook_fn
        """
        assert model is not None, "no model found"
        self.model = model
        self.step = 0
        self.activations = {}
        self.hooks = []
        self.activations_to_cache = []
        for name, _ in model.named_modules():
            # check for regex match
            if any(regex.match(activation, name) for activation in activation_regex):
                self.activations_to_cache.append(name)

        # Natural sort to impose a consistent order
        self.activations_to_cache = natsorted(self.activations_to_cache)
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
            if name in self.activations_to_cache:
                hook_fn = self.hook_fn if self.hook_fn is not None else self._get_caching_hook(
                    name)
                forward_hook = module.register_forward_hook(hook_fn)
                self.hooks.append(forward_hook)

    def _get_caching_hook(self, name):
        def hook(module, input, output):
            output_ = output.detach().cpu()
            self.activations[f"{name}"] = output_

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
        self.activations = {}


class WhisperActivationCache(BaseActivationModule):
    """
    Use hooks in BaseActivationModule to cache intermediate activations while running forward pass
    """

    def __init__(
        self,
        hook_fn: Optional[Callable] = None,
        model: Optional[torch.nn.Module] = None,
        activation_regex: list[str] = ["*.blocks.*.mlp.*"],
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(model, hook_fn, activation_regex)
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
            if "decoder" in name:
                # we don't cache the first activations that correspond to the sos/lang tokens
                if output.shape[1] > 1:
                    del self.activations[f"{name}"]
                    return
            output_ = output.detach().cpu()
            if name in self.activations:
                self.activations[f"{name}"] = torch.cat(
                    (self.activations[f"{name}"], output_), dim=1
                )
            else:
                self.activations[f"{name}"] = output_

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
            return substitution_activation

        return hook
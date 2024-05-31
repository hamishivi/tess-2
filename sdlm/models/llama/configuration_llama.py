"""Adapted Llama configuration for diffusion models."""

from transformers.models.llama.configuration_llama import LlamaConfig

from sdlm.models.mixins.configuration_mixin import DiffusionConfigMixin


class LlamaDiffusionConfig(DiffusionConfigMixin, LlamaConfig):
    def __init__(self, *args, **kwargs):
        LlamaConfig.__init__(self, *args, **kwargs)
        DiffusionConfigMixin.__init__(self, *args, **kwargs)

"""Adapted Mistral configuration for diffusion models."""

from transformers.models.mistral import MistralConfig

from sdlm.models.mixins.configuration_mixin import DiffusionConfigMixin


class MistralDiffusionConfig(DiffusionConfigMixin, MistralConfig):
    def __init__(self, *args, **kwargs):
        MistralConfig.__init__(self, *args, **kwargs)
        DiffusionConfigMixin.__init__(self, *args, **kwargs)

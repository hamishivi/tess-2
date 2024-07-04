"""Adapted Mistral configuration for diffusion models."""

from transformers.models.mistral import MistralConfig

from sdlm.models.mixins.configuration_mixin import DiffusionConfigMixin


class MistralDiffusionConfig(DiffusionConfigMixin, MistralConfig):
    def __init__(self, *args, **kwargs):
        MistralConfig.__init__(self, *args, **kwargs)
        DiffusionConfigMixin.__init__(self, *args, **kwargs)


class CDCDMistralDiffusionConfig(MistralDiffusionConfig):
    def __init__(self, *args, n_bins: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bins = n_bins

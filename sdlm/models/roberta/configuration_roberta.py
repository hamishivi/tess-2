"""Adapted Roberta configuration for diffusion models."""

from transformers.models.roberta.configuration_roberta import RobertaConfig

from sdlm.models.mixins.configuration_mixin import DiffusionConfigMixin


class RobertaDiffusionConfig(DiffusionConfigMixin, RobertaConfig):
    def __init__(self, *args, **kwargs):
        RobertaConfig.__init__(self, *args, **kwargs)
        DiffusionConfigMixin.__init__(self, *args, **kwargs)

"""Adapted Roberta configuration for diffusion models."""

from transformers.models.roberta.configuration_roberta import RobertaConfig
from typing import Optional

class RobertaDiffusionConfig(RobertaConfig):
    def __init__(self, self_condition: Optional[str]= None, **kwargs):
        super().__init__(**kwargs)
        self.self_condition = self_condition

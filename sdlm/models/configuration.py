"""Adapted Roberta configuration for diffusion models."""

from transformers.models.roberta.configuration_roberta import RobertaConfig
from typing import Optional

class RobertaDiffusionConfig(RobertaConfig):
    def __init__(self, self_condition: Optional[str]= None, self_condition_zeros_after_softmax: bool=False,  deepmind_conditional: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.self_condition = self_condition
        self.self_condition_zeros_after_softmax = self_condition_zeros_after_softmax
        self.deepmind_conditional = deepmind_conditional

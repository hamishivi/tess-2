"""Adapted XLM Roberta configuration for diffusion models."""

from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from typing import Optional


class XLMRobertaDiffusionConfig(XLMRobertaConfig):
    def __init__(
        self,
        self_condition: Optional[str] = None,
        self_condition_zeros_after_softmax: bool = False,
        deepmind_conditional: bool = False,
        classifier_free_simplex_inputs: bool = False,
        self_condition_mlp_projection=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.self_condition = self_condition
        self.self_condition_zeros_after_softmax = self_condition_zeros_after_softmax
        self.deepmind_conditional = deepmind_conditional
        self.classifier_free_simplex_inputs = classifier_free_simplex_inputs
        self.self_condition_mlp_projection = self_condition_mlp_projection

"""Adapted Roberta configuration for diffusion models."""

from transformers.models.roberta.configuration_roberta import RobertaConfig
from typing import Optional


class RobertaDiffusionConfig(RobertaConfig):
    def __init__(
        self,
        self_condition: Optional[str] = None,
        self_condition_zeros_after_softmax: bool = False,
        deepmind_conditional: bool = False,
        classifier_free_simplex_inputs: bool = False,
        classifier_free_uncond_input: str = "empty_token",
        self_condition_mlp_projection=False,
        self_condition_mix_before_weights=False,
        self_condition_mix_logits_before_weights=False,
        empty_token_be_mask=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_condition = self_condition
        self.self_condition_zeros_after_softmax = self_condition_zeros_after_softmax
        self.deepmind_conditional = deepmind_conditional
        self.classifier_free_simplex_inputs = classifier_free_simplex_inputs
        self.classifier_free_uncond_input = classifier_free_uncond_input
        self.self_condition_mlp_projection = self_condition_mlp_projection
        self.self_condition_mix_before_weights = self_condition_mix_before_weights
        self.self_condition_mix_logits_before_weights = self_condition_mix_logits_before_weights
        self.empty_token_be_mask=empty_token_be_mask
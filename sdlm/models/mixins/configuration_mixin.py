from typing import Optional


class DiffusionConfigMixin:
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
        is_causal: bool = False,
        mask_padding_in_loss: bool = False,
        padding_side: str = "right",
        **kwargs,
    ):
        self.self_condition = self_condition
        self.self_condition_zeros_after_softmax = self_condition_zeros_after_softmax
        self.deepmind_conditional = deepmind_conditional
        self.classifier_free_simplex_inputs = classifier_free_simplex_inputs
        self.classifier_free_uncond_input = classifier_free_uncond_input
        self.self_condition_mlp_projection = self_condition_mlp_projection
        self.self_condition_mix_before_weights = self_condition_mix_before_weights
        self.self_condition_mix_logits_before_weights = (
            self_condition_mix_logits_before_weights
        )
        self.empty_token_be_mask = empty_token_be_mask
        self.is_causal = is_causal
        self.mask_padding_in_loss = mask_padding_in_loss
        self.padding_side = padding_side

from typing import Optional, Tuple, Union

import torch
import pdb
from diffusers.pipeline_utils import DiffusionPipeline
from sdlm.inference.inference_utils import logits_projection
from sdlm.utils import scale, self_condition_preds
from dataclasses import dataclass
import numpy as np
from diffusers.utils import BaseOutput
from sdlm.utils import convert_to_simplex
import torch.nn.functional as F

@dataclass
class SimplexDiffusionPipelineOutput(BaseOutput):
    """
    Output class for simplex diffusion pipelines.
    Args:
        simplex (`np.ndarray`)
            numpy array showing the denoised simplex representation.
        logits (`np.ndarray`) final generated logits before applying the projection.
    """

    simplex: np.ndarray
    logits: np.ndarray
    loss: np.ndarray


class SimplexDDPMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Parameters:
        model: Model architecture to denoise the latents (encoded token ids).
        scheduler ([`SchedulerMixin`]): A scheduler to denoise the encoded latent.
    """

    def __init__(
        self,
        model,
        scheduler,
        simplex_value,
        top_p,
        sampling_type,
        is_conditional_generation,
        tokenizer,
        classifier_free_uncond_input,
        temperature,
        guidance_softmax_combination
    ):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        self.simplex_value = simplex_value
        self.top_p = top_p
        self.sampling_type = sampling_type
        self.is_conditional_generation = is_conditional_generation
        self.tokenizer = tokenizer
        self.classifier_free_uncond_input = classifier_free_uncond_input
        self.temperature = temperature
        self.guidance_softmax_combination=guidance_softmax_combination

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        seq_length: int = 512,
        generator: Optional[torch.Generator] = None,
        batch: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 1.0,
    ) -> Union[SimplexDiffusionPipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            seq_length: (`int`), sequence length for the generated samples.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            batch (`torch.FloatTensor`): batch of input data, mostly used in the conditional generation setting.
        Returns:
            [`~pipeline_utils.SimplexDiffusionPipelineOutput`]: returns the generated simplex.
        """
        # Classifier_free guidance works only in the conditional generation case.
        classifier_free_guidance = guidance_scale > 1.0 and self.is_conditional_generation
        """
        if classifier_free_guidance:
            # Makes unconditional input for max sequence length, later we truncate it.
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=seq_length, return_tensors="pt"
            ).to(self.device)
            # Converts this to a simplex (batch_size, max_seq, vocab_size)
            uncond_simplex = convert_to_simplex(uncond_input["input_ids"], self.simplex_value, self.model.config.vocab_size)
        """
        # Sample gaussian noise to begin loop
        vocab_size = self.model.config.vocab_size
        if batch is not None:
            # TODO(rabeeh): is giving the length cheating for this setting?
            # Adapts the sequence length to the given `span_mask`'s length.
            seq_length = batch["input_ids"].shape[1]
        simplex_shape = (batch_size, seq_length, vocab_size)
        simplex = self.simplex_value * torch.randn(simplex_shape, generator=generator, device=self.device)
        if self.model.config.self_condition is not None:
            previous_pred = torch.zeros((batch_size, seq_length, vocab_size), device=self.device)
        logits_projection_fct = lambda x: logits_projection(
            x, self.sampling_type, self.top_p, self.simplex_value, self.temperature
        )

        for t in self.progress_bar(self.scheduler.timesteps):
            # TODO(rabeeh): also check without the scale.
            t_scaled = scale(t, len(self.scheduler))
            """
            if classifier_free_guidance:
                if self.classifier_free_uncond_input == "empty_token":
                    uncond_input = uncond_simplex[:, : batch["input_ids"].shape[1], :]
                elif self.classifier_free_uncond_input == "noisy_simplex":
                    uncond_input = self.simplex_value * torch.randn(simplex.shape, generator=generator, device=self.device)
                else:
                    raise NotImplementedError
            """
            # 1. predict noise model_output. Note we need not to pass the input_ids in case of
            # unconditional generation since the loss would be computed and it should not.
            model_output = self.model(
                input_ids=batch["input_ids"] if self.is_conditional_generation else None,
                span_mask=batch["span_mask"] if self.is_conditional_generation else None,
                simplex=simplex,
                timesteps=t_scaled,
                previous_pred=previous_pred if self.model.config.self_condition else None,
                classifier_free_guidance=classifier_free_guidance,
                # unconditional_simplex=uncond_input if classifier_free_guidance else None,
            )
            model_output_logits = model_output.logits

            # Performs classifier-free guidance.
            if classifier_free_guidance:
                logits_uncond, logits_pred = model_output_logits.chunk(2)
                if self.guidance_softmax_combination:
                    model_output_logits = F.softmax(logits_uncond, dim=-1) +  guidance_scale * (F.softmax(logits_pred, dim=-1) - F.softmax(logits_uncond, dim=-1))
                else:
                    model_output_logits = logits_uncond + guidance_scale * (logits_pred - logits_uncond)

            if self.model.config.self_condition is not None:
                if classifier_free_guidance:
                    prev_output_logits = model_output.logits.chunk(2)[1]
                else:
                    prev_output_logits = model_output_logits

                previous_pred = self_condition_preds(
                    self.model.config.self_condition, prev_output_logits, logits_projection_fct
                )

            # Projection.
            projected_logits = logits_projection_fct(model_output_logits)

            # 2. compute previous logits: x_t -> x_t-1
            noise = self.simplex_value * torch.randn(simplex_shape, generator=generator, device=self.device)
            simplex = self.scheduler.step(projected_logits, t, noise, generator=generator).prev_sample

        return SimplexDiffusionPipelineOutput(simplex=simplex, logits=model_output_logits, loss=model_output.loss)

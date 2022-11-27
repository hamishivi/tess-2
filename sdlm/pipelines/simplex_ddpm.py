from typing import Optional, Tuple, Union

import torch
import pdb
from diffusers.pipeline_utils import DiffusionPipeline
from sdlm.inference.inference_utils import sample_logits
from sdlm.utils import convert_to_simplex, scale
from dataclasses import dataclass
import numpy as np
from diffusers.utils import BaseOutput


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



class SimplexDDPMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Parameters:
        model: Model architecture to denoise the latents (encoded token ids).
        scheduler ([`SchedulerMixin`]): A scheduler to denoise the encoded latent.
    """

    def __init__(self, model, scheduler, simplex_value, top_p, sampling_type, span_infilling):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        self.simplex_value = simplex_value
        self.top_p = top_p
        self.sampling_type = sampling_type
        self.span_infilling = span_infilling

    def logits_projection(self, logits):
        # TODO(rabeeh): huggingface has different sampling, like constrastive one.
        # also there are more variant in diffusion-lm.
        token_ids = sample_logits(self.sampling_type, logits, self.top_p)
        return convert_to_simplex(token_ids, self.simplex_value, vocab_size=logits.shape[2])

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        seq_length: int = 512,
        generator: Optional[torch.Generator] = None,
        batch: Optional[torch.FloatTensor] = None,
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
        # Sample gaussian noise to begin loop
        vocab_size = self.model.config.vocab_size
        if batch is not None:
            # TODO(rabeeh): is giving the length cheating for this setting?
            # Adapts the sequence length to the given `span_mask`'s length.
            seq_length = batch["input_ids"].shape[1]
        simplex_shape = (batch_size, seq_length, vocab_size)
        simplex = self.simplex_value * torch.randn(simplex_shape, generator=generator, device=self.device)
        if self.model.config.self_condition == "hidden_state":
            previous_pred = torch.zeros((batch_size, seq_length, self.model.config.hidden_size), device=self.device)
        for t in self.progress_bar(self.scheduler.timesteps):
            # TODO(rabeeh): also check without the scale.
            t_scaled = scale(t, len(self.scheduler))
            # 1. predict noise model_output
            model_output = self.model(simplex=simplex,
                timesteps=t_scaled, 
                input_ids=batch["input_ids"] if self.span_infilling else None,
                span_mask=batch["span_mask"] if self.span_infilling else None,
                previous_pred=previous_pred if self.model.config.self_condition else None)
            
            if self.model.config.self_condition == "hidden_state":
                previous_pred = model_output.hidden_states
            
            # Projection.
            projected_logits = self.logits_projection(model_output.logits)

            # 2. compute previous logits: x_t -> x_t-1
            noise = self.simplex_value * torch.randn(simplex_shape, generator=generator, device=self.device)
            simplex = self.scheduler.step(projected_logits, t, noise, generator=generator).prev_sample

        return SimplexDiffusionPipelineOutput(simplex=simplex, logits=model_output.logits)

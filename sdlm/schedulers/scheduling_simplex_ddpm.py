"""DDPM scheduler for the simplex diffusion model."""

from diffusers import DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from dataclasses import dataclass
from typing import Union, Tuple, Optional
import torch
import numpy as np
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
import math 

@dataclass
class SimplexDDPMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.
    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        projected_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            The projected logits sample (x_{0}) based on the model output from the current timestep.
    """

    prev_sample: torch.FloatTensor
    projected_logits: Optional[torch.FloatTensor] = None


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999, improved_ddpm=False):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].
    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.
    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def default_alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2
    
    if improved_ddpm:
        # Implements eqn. 17 in https://arxiv.org/pdf/2102.09672.pdf. 
        alpha_bar = lambda x: (default_alpha_bar(x)/default_alpha_bar(0.0))
        alphas_cumprod= []
    else:
        alpha_bar = default_alpha_bar
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        alpha_bar_t1 = alpha_bar(t1)
        betas.append(min(1 - alpha_bar(t2) / alpha_bar_t1, max_beta))
        if improved_ddpm:
           alphas_cumprod.append(alpha_bar_t1)
    betas = torch.tensor(betas, dtype=torch.float32)
    if improved_ddpm:
        return betas, torch.tensor(alphas_cumprod, dtype=torch.torch.float32)  
    return betas

class SimplexDDPMScheduler(DDPMScheduler):
    
    @register_to_config
    def __init__(
        self,
        simplex_value: float,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
    ):
        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "squaredcos_improved_ddpm":
            self.betas, self.alphas_cumprod = betas_for_alpha_bar(num_train_timesteps, improved_ddpm=True)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        if beta_schedule == "squaredcos_improved_ddpm":
            self.alphas = None 
        else:
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        self.variance_type = variance_type

    def step(
        self,
        projected_logits: torch.FloatTensor,
        timestep: int,
        noise: torch.FloatTensor,
        generator=None,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        Args:
            projected_logits (`torch.FloatTensor`): projected logits from the diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            noise (`torch.FloatTensor`): a random noise with simplex_value standard deviation.
            generator: random number generator.
        Returns:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] resulted values.
        """
        t = timestep

        # 1. compute alphas, betas
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one
        
        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            projected_logits = torch.clamp(projected_logits, -1, 1)

        # See algorithm 2 in Figure 3 in https://arxiv.org/pdf/2210.17432.pdf.
        predicted_logits_coeff = alpha_prod_t_prev ** (0.5)
        noise_coeff = (1-alpha_prod_t_prev) ** (0.5)
        pred_prev_sample = predicted_logits_coeff * projected_logits + noise_coeff * noise

        return SimplexDDPMSchedulerOutput(prev_sample=pred_prev_sample, projected_logits=projected_logits)


    
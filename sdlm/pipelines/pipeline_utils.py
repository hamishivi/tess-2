"""Defines utilities for diffusion pipelines."""
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
    """

    simplex: np.ndarray


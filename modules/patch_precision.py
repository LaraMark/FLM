# Consistent with Kohya to reduce differences between model training and inference.

import torch
import math
import einops
import numpy as np

# Import necessary modules from the ldm_patched package
import ldm_patched.ldm.modules.diffusionmodules.openaimodel
import ldm_patched.modules.model_sampling
import ldm_patched.modules.sd1_clip

# Import a utility function for creating beta schedules
from ldm_patched.ldm.modules.diffusionmodules.util import make_beta_schedule

def patched_timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    # Consistent with Kohya to reduce differences between model training and inference.

    # Check if we should only repeat the timesteps tensor along the specified dimension
    if repeat_only:
        # Repeat the timesteps tensor along the specified dimension
        embedding = einops.repeat(timesteps, 'b -> b d', d=dim)
    else:
        # Calculate the half of the specified dimension
        half = dim // 2

        # Create a range of frequencies from 0 to half, with the specified max_period
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)

        # Calculate the arguments for the cosine and sine functions
        args = timesteps[:, None].float() * freqs[None]

        # Create the embedding by concatenating the cosine and sine values along the last dimension
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # If the dimension is odd, add a zero tensor along the last dimension
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding

def patched_register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    # Consistent with Kohya to reduce differences between model training and inference.

    # If the user provides a custom beta schedule, use it
    if given_betas is not None:
        betas = given_betas
    else:
        # Otherwise, create a beta schedule based on the specified parameters
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s)

    # Calculate the corresponding alphas and cumulative product of alphas
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    # Store the number of timesteps, linear start and end values
    timesteps, = betas.shape
    self.num_timesteps = int(timesteps)
    self.linear_start = linear_start
    self.linear_end = linear_end

    # Calculate the standard deviation at each timestep
    sigmas = torch.tensor(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, dtype=torch.float32)

    # Set the sigmas attribute of the ModelSamplingDiscrete instance
    self.set_sigmas(sigmas)
    return

def patch_all_precision():
    # Replace the timestep_embedding function in the openaimodel module
    ldm_patched.ldm.modules.diffusionmodules.openaimodel.timestep_embedding = patched_timestep_embedding

    # Replace the _register_schedule method in the ModelSamplingDiscrete class
    ldm_patched.modules.model_sampling.ModelSamplingDiscrete._register_schedule = patched_register_schedule
    return

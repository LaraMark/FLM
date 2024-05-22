# Import necessary libraries and modules
import torch
from torch import einsum
import torch.nn.functional as F
import math

from einops import rearrange, repeat
import os
from ldm_patched.ldm.modules.attention import optimized_attention, _ATTN_PRECISION
import ldm_patched.modules.samplers

# Function to perform attention with similarity scores
def attention_basic_with_sim(q, k, v, heads, mask=None):
    # Calculate the shape of q and set the scale factor
    b, _, dim_head = q.shape
    dim_head //= heads
    scale = dim_head ** -0.5

    # Perform reshaping, unsqueezing, and permutation operations on q, k, and v
    h = heads
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    # Calculate the similarity scores between q and k
    sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale

    # If mask is not None, apply the mask to the similarity scores
    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # Perform softmax on the similarity scores
    sim = sim.softmax(dim=-1)

    # Calculate the output using the similarity scores and v
    out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return (out, sim)

# Function to create a blur map
def create_blur_map(x0, attn, sigma=3.0, threshold=1.0):
    # Reshape and apply Global Average Pooling to the attention map
    _, hw1, hw2 = attn.shape
    b, _, lh, lw = x0.shape
    attn = attn.reshape(b, -1, hw1, hw2)
    mask = attn.mean(1, keepdim=False).sum(1, keepdim=False) > threshold

    # Calculate the ratio and mid_shape
    ratio = math.ceil(math.sqrt(lh * lw / hw1))
    mid_shape = [math.ceil(lh / ratio), math.ceil(lw / ratio)]

    # Reshape, unsqueeze, and interpolate the mask
    mask = (
        mask.reshape(b, *mid_shape)
        .unsqueeze(1)
        .type(attn.dtype)
    )
    mask = F.interpolate(mask, (lh, lw))

    # Perform Gaussian blur on x0 and combine it with the original x0 using the mask
    blurred = gaussian_blur_2d(x0, kernel_size=9, sigma=sigma)
    blurred = blurred * mask + x0 * (1 - mask)
    return blurred

# Function to perform Gaussian blur on a 2D image
def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    # Create the x kernel
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    x_kernel = pdf / pdf.sum()

    # Create the 2D kernel
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)
    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    # Perform padding and convolution
    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])
    return img

# SelfAttentionGuidance class
class SelfAttentionGuidance:
    # Define the input types
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "scale": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 5.0, "step": 0.1}),
                             "blur_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                              }}
    # Define the return types
    RETURN_TYPES = ("MODEL",)
    # Define the function name
    FUNCTION = "patch"

    # Set the category
    CATEGORY = "_for_testing"

    # Patch function
    def patch(self, model, scale, blur_sigma):
        # Create a clone of the model
        m = model.clone()

        # Initialize attn_scores
        attn_scores = None

        # Define the attention and recording function
        def attn_and_record(q, k, v, extra_options):
            # Implement the logic for saving attention scores when using unconditional mode
            heads = extra_options["n_heads"]
            cond_or_uncond = extra_options["cond_or_uncond"]
            b = q.shape[0] // len(cond_or_uncond)
            if 1 in cond_or_uncond:
                uncond_index = cond_or_uncond.index(1)
                # Perform the entire attention operation and save the attention scores
                (out, sim) = attention_basic_with_sim(q, k, v, heads=heads)
                # When using a higher batch size, the result batch dimension is [uc1, ... ucn, c1, ... cn]
                n_slices = heads * b
                attn_scores = sim[n_slices * uncond_index:n_slices * (uncond_index+1)]
                return out
            else:
                # Otherwise, use the optimized attention function
                return optimized_attention(q, k, v, heads=heads)

        # Define the post-configuration function
        def post_cfg_function(args):
            # Implement the logic for creating the adversarially blurred image and applying the guidance scale
            nonlocal attn_scores
            uncond_attn = attn_scores

            sag_scale = scale
            sag_sigma = blur_sigma
            sag_threshold = 1.0
            model = args["model"]
            uncond_pred = args["uncond_denoised"]
            uncond = args["uncond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"]
            x = args["input"]

            # Create the adversarially blurred image
            degraded = create_blur_map(uncond_pred, uncond_attn, sag_sigma, sag_threshold)
            degraded_noised = degraded + x - uncond_pred
            # Call into the UNet
            (sag, _) = ldm_patched.modules.samplers.calc_cond_uncond_batch(model, uncond, None, degraded_noised, sigma, model_options)
            return cfg_result + (degraded - sag) * sag_scale

        # Set the model's post-configuration function
        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)

        # Replace the attention function in the UNet
        m.set_model_attn1_replace(attn_and_record, "middle", 0, 0)

        # Return the modified model
        return (m, )

# Define the node class mappings
NODE_CLASS_MAPPINGS = {
    "SelfAttentionGuidance": SelfAttentionGuidance,
}

# Define the node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfAttentionGuidance": "Self-Attention Guidance",
}

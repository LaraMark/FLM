# Import necessary modules and classes
import torch
import ldm_patched.contrib.external
import ldm_patched.modules.utils

def camera_embeddings(elevation, azimuth):
    """
    Generate camera embeddings for a given elevation and azimuth.

    Args:
    elevation (torch.Tensor): Elevation value.
    azimuth (torch.Tensor): Azimuth value.

    Returns:
    torch.Tensor: Embeddings for the given elevation and azimuth.
    """
    elevation = torch.as_tensor([elevation])
    azimuth = torch.as_tensor([azimuth])
    embeddings = torch.stack(
        [
            torch.deg2rad(
                (90 - elevation) - (90)
            ),  # Zero123 polar is 90-elevation
            torch.sin(torch.deg2rad(azimuth)),
            torch.cos(torch.deg2rad(azimuth)),
            torch.deg2rad(
                90 - torch.full_like(elevation, 0)
            ),
        ], dim=-1).unsqueeze(1)

    return embeddings


class StableZero123_Conditioning:
    @classmethod
    def INPUT_TYPES(s):
        """
        Define the required input types for the StableZero123_Conditioning class.

        Returns:
        dict: A dictionary containing the required input types.
        """
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "init_image": ("IMAGE",),
                              "vae": ("VAE",),
                              "width": ("INT", {"default": 256, "min": 16, "max": ldm_patched.contrib.external.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 256, "min": 16, "max": ldm_patched.contrib.external.MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "elevation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                              "azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                             }}

    # Define the return types, names, and function for the class
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/3d_models"

    def encode(self, clip_vision, init_image, vae, width, height, batch_size, elevation, azimuth):
        """
        Encode the input image and camera embeddings to generate conditioning and latent variables.

        Args:
        clip_vision (CLIP_VISION): A class instance for CLIP vision.
        init_image (IMAGE): The initial image to be encoded.
        vae (VAE): A class instance for Variational Autoencoder (VAE).
        width (INT): The width of the output image.
        height (INT): The height of the output image.
        batch_size (INT): The batch size for the output.
        elevation (FLOAT): The elevation value for the camera embeddings.
        azimuth (FLOAT): The azimuth value for the camera embeddings.

        Returns:
        tuple: A tuple containing positive and negative conditioning and latent variables.
        """
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = ldm_patched.modules.utils.common_upscale(init_image.movedim(-1,1), width, height, "bilinear", "center").movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        t = vae.encode(encode_pixels)
        cam_embeds = camera_embeddings(elevation, azimuth)
        cond = torch.cat([pooled, cam_embeds.repeat((pooled.shape[0], 1, 1))], dim=-1)

        positive = [[cond, {"concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled), {"concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return (positive, negative, {"samples":latent})

NODE_CLASS_MAPPINGS = {
    "StableZero123_Conditioning": StableZero123_Conditioning,
}

# Import necessary libraries and modules
import torch
from . import model_base  # Base class for all models
from . import utils  # Utility functions

# Import model-specific modules
from . import sd1_clip, sd2_clip, sdxl_clip  # CLIP models for different Stable Diffusion versions
from . import supported_models_base  # Base class for supported models
from . import latent_formats  # Different latent formats for Stable Diffusion models
from . import diffusers_convert  # Utility functions for converting state dictionaries

class SD15(supported_models_base.BASE):
    # Configuration for the U-Net model
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    # Additional configuration for the U-Net model
    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    # Define the latent format for this model
    latent_format = latent_formats.SD15

    # Method to process the CLIP state dictionary
    def process_clip_state_dict(self, state_dict):
        # Replace prefixes in the state dictionary keys
        k = list(state_dict.keys())
        for x in k:
            if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
                y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
                state_dict[y] = state_dict.pop(x)

        # Round position IDs in the state dictionary
        if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in state_dict:
            ids = state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids']
            if ids.dtype == torch.float32:
                state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

        # Replace prefixes in the state dictionary keys for specific keys
        replace_prefix = {}
        replace_prefix["cond_stage_model."] = "cond_stage_model.clip_l."
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix)

        return state_dict

    # Method to process the CLIP state dictionary for saving
    def process_clip_state_dict_for_saving(self, state_dict):
        # Replace prefixes in the state dictionary keys for specific keys
        replace_prefix = {"clip_l.": "cond_stage_model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    # Method to get the CLIP target for this model
    def clip_target(self):
        return supported_models_base.ClipTarget(sd1_clip.SD1Tokenizer, sd1_clip.SD1ClipModel)

# Repeat the above pattern for other models (SD20, SD21UnclipL, SD21UnclipH, SDXLRefiner, SDXL, SSD1B, Segmind_Vega, SVD_img2vid, Stable_Zero123)

# Define a list of all supported models
models = [Stable_Zero123, SD15, SD20, SD21UnclipL, SD21UnclipH, SDXLRefiner, SDXL, SSD1B, Segmind_Vega]
models += [SVD_img2vid]

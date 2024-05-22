# https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py 

import torch
from ldm_patched.contrib.external import MAX_RESOLUTION

class CLIPTextEncodeSDXLRefiner:
    """
    A class that encodes text input using the CLIP model for conditioning in SDXL.
    This refined version allows for an aesthetic score to be included.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Define the required input types for the encode function.
        """
        return {
            "required": {
                "ascore": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"min": 0, "max": MAX_RESOLUTION}),
                "text": ("STRING", {"multiline": True}),  # Multiline text input
                "clip": ("CLIP", ),                      # CLIP model instance
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, ascore, width, height, text):
        """
        Encode text input using the CLIP model for conditioning in SDXL.

        :param clip: CLIP model instance
        :param ascore: Aesthetic score for the input
        :param width: Width of the input image
        :param height: Height of the input image
        :param text: Text input to be encoded
        :return: Encoded conditioning data
        """
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled, "aesthetic_score": ascore, "width": width, "height": height}]], )

class CLIPTextEncodeSDXL:
    """
    A class that encodes text input using the CLIP model for conditioning in SDXL.
    This version supports multiple text inputs and cropping.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Define the required input types for the encode function.
        """
        return {
            "required": {
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"min": 0, "max": MAX_RESOLUTION}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "text_g": ("STRING", {"multiline": True, "default": "CLIP_G"}),  # Multiline text input
                "clip": ("CLIP", ),                                             # CLIP model instance
                "text_l": ("STRING", {"multiline": True, "default": "CLIP_L"}),  # Multiline text input
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l):
        """
        Encode text input using the CLIP model for conditioning in SDXL.

        :param clip: CLIP model instance
        :param width: Width of the input image
        :param height: Height of the input image
        :param crop_w: Width of the crop

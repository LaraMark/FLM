# https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py 

import torch
from ldm_patched.contrib.external import MAX_RESOLUTION

class CLIPTextEncodeSDXLRefiner:
    """
    A class that encodes text input using CLIP text encoder and returns a conditioning dictionary.
    This version includes an aesthetic score attribute.
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
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "text": ("STRING", {"multiline": True}),  # Accepts multiline strings
                "clip": ("CLIP", ),  # Requires a CLIP instance
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, ascore, width, height, text):
        """
        Encode text input using CLIP text encoder and return a conditioning dictionary with an aesthetic score.

        :param clip: A CLIP instance
        :param ascore: Aesthetic score for the generated image
        :param width: Width of the generated image
        :param height: Height of the generated image
        :param text: Text input to be encoded
        :return: A tuple containing the encoded conditioning dictionary
        """
        tokens = clip.tokenize(text)  # Tokenize the input text
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)  # Encode the tokens
        return ([[cond, {"pooled_output": pooled, "aesthetic_score": ascore, "width": width, "height": height}]], )  # Return the encoded conditioning dictionary

class CLIPTextEncodeSDXL:
    """
    A class that encodes text input using CLIP text encoder and returns a conditioning dictionary.
    This version includes cropping and target resolution attributes.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Define the required input types for the encode function.
        """
        return {
            "required": {
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "text_g": ("STRING", {"multiline": True, "default": "CLIP_G"}),  # Accepts multiline strings
                "text_l": ("STRING", {"multiline": True, "default": "CLIP_L"}),  # Accepts multiline strings
                "clip": ("CLIP", ),  # Requires a CLIP instance
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l):
        """
        Encode text input using CLIP text encoder and return a conditioning dictionary with cropping and target resolution attributes.

        :param clip: A CLIP instance
        :param width: Width of the generated image
        :param height: Height of the generated image
        :param crop_w: Width of the cropping rectangle
        :param crop_h: Height of the cropping rectangle
        :param target_width: Target width of the generated image
        :param target_height: Target height of the generated image
        :param text_g: Global text input to be encoded
        :param text_l: Local text input to be encoded
        :return: A tuple containing the encoded conditioning dictionary
        """
        tokens = clip.tokenize(text_g)  # Tokenize the global text input
        tokens["l"] = clip.tokenize(text_l)["l"]  # Tokenize the local text input
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")  # Tokenize an empty string
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]  # Pad the local text input if it's shorter
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]  # Pad the global text input if it's shorter
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)  # Encode the tokens
        return ([[cond, {"pooled_output": pooled, "width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]], )  # Return the encoded conditioning dictionary

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeSDXLRefiner": CLIPTextEncodeSDXLRefiner,
    "CLIPTextEncodeSDXL": CLIPTextEncodeSDXL,
}

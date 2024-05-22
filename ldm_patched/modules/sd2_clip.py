# Import necessary modules
from ldm_patched.modules import sd1_clip
import torch
import os

# Define a class SD2ClipHModel that inherits from sd1_clip.SDClipModel
class SD2ClipHModel(sd1_clip.SDClipModel):
    def __init__(self, arch="ViT-H-14", device="cpu", max_length=77, freeze=True, layer="penultimate", layer_idx=None, dtype=None):
        # If the layer is set to "penultimate", change it to "hidden" and set layer_idx to -2
        if layer == "penultimate":
            layer="hidden"
            layer_idx=-2

        # Initialize the superclass with the given parameters
        textmodel_json_config = os os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd2_clip_config.json")
        super().__init__(device=device, freeze=freeze, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"start": 49406, "end": 49407, "pad": 0})

# Define a class SD2ClipHTokenizer that inherits from sd1_clip.SDTokenizer
class SD2ClipHTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, tokenizer_path=None, embedding_directory=None):
        # Initialize the superclass with the given parameters
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=1024)

# Define a class SD2Tokenizer that inherits from sd1_clip.SD1Tokenizer
class SD2Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None):
        # Initialize the superclass with the given parameters
        super().__init__(embedding_directory=embedding_directory, clip_name="h", tokenizer=SD2ClipHTokenizer)

# Define a class SD2ClipModel that inherits from sd1_clip.SD1ClipModel
class SD2ClipModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, **kwargs):
        # Initialize the superclass with the given parameters
        super().__init__(device=device, dtype=dtype, clip_name="h", clip_model=SD2ClipHModel, **kwargs)

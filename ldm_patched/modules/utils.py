import torch
import math
import struct
import ldm_patched.modules.checkpoint_pickle
import safetensors.torch
import numpy as np
from PIL import Image

def load_torch_file(ckpt, safe_load=False, device=None):
    # This function loads a torch file (.pth or .safetensors) and returns the state dictionary.
    # If safe_load is True, it will use the safe_load function from safetensors.torch to load the file.
    # If the file is not a .safetensors file, it will use torch.load with the provided device and
    # optionally load only the weights (if safe_load is True).

def save_torch_file(sd, ckpt, metadata=None):
    # This function saves a state dictionary to a torch file (.pth or .safetensors).
    # If metadata is not None, it will save the metadata along with the state dictionary.

def calculate_parameters(sd, prefix=""):
    # This function calculates the total number of parameters in the state dictionary
    # with the given prefix.

def state_dict_key_replace(state_dict, keys_to_replace):
    # This function replaces keys in the state dictionary with new keys.

def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    # This function replaces a prefix in the state dictionary keys with a new prefix.

def transformers_convert(sd, prefix_from, prefix_to, number):
    # This function converts a state dictionary from the Hugging Face Transformers format
    # to the Diffusers format.

UNET_MAP_ATTENTIONS = {...}
TRANSFORMER_BLOCKS = {...}
UNET_MAP_RESNET = {...}
UNET_MAP_BASIC = {...}

def unet_to_diffusers(unet_config):
    # This function converts a U-Net configuration to a Diffusers configuration.

def repeat_to_batch_size(tensor, batch_size):
    # This function repeats or truncates the tensor to have a batch size of batch_size.

def resize_to_batch_size(tensor, batch_size):
    # This function resizes the tensor to have a batch size of batch_size.

def convert_sd_to(state_dict, dtype):
    # This function converts the data type of all tensors in the state dictionary to the given dtype.

def safetensors_header(safetensors_path, max_size=100*1024*1024):
    # This function reads the header of a .safetensors file and returns it.

def set_attr(obj, attr, value):
    # This function sets an attribute of an object to a new value.

def copy_to_param(obj, attr, value):
    # This function copies a tensor to a parameter of an object.

def get_attr(obj, attr):
    # This function gets an attribute of an object.

def bislerp(samples, width, height):
    # This function performs bi-linear interpolation on a batch of images.

def lanczos(samples, width, height):
    # This function upscales a batch of images using the Lanczos resampling algorithm.

def common_upscale(samples, width, height, upscale_method, crop):
    # This function upscales a batch of images using a given upscale method and crop option.

def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    # This function calculates the number of tiles needed to cover the input image
    # with the given tile size and overlap.

@torch.inference_mode()
def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu", pbar = None):
    # This function applies a function to a batch of images in a tiled and parallelized manner.

PROGRESS_BAR_ENABLED = True
def set_progress_bar_enabled(enabled):
    # This function enables or disables the global progress bar.

PROGRESS_BAR_HOOK = None
def set_progress_bar_global_hook(function):
    # This function sets the global progress bar hook function.

class ProgressBar:
    # This class implements a simple progress bar.
    def __init__(self, total):
        ...
    def update_absolute(self, value, total=None, preview=None):
        ...
    def update(self, value):
        ...

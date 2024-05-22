import torch
from torch import nn
from ldm_patched.ldm.modules.attention import CrossAttention  # Importing CrossAttention


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


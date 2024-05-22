import torch
import ldm_patched.modules.clip_vision  # Importing ClipVisionModel
import safetensors.torch as sf  # Importing safetensors for torch
import ldm_patched.modules.model_management as model_management  # Importing model management functions
import ldm_patched.ldm.modules.attention as attention  # Importing attention functions

from extras.resampler import Resampler  # Importing Resampler class
from ldm_patched.modules.model_patcher import ModelPatcher  # Importing ModelPatcher class
from modules.core import numpy_to_pytorch  # Importing numpy_to_pytorch function
from modules.ops import use_patched_ops  # Importing use_patched_ops function
from ldm_patched.modules.ops import manual_cast  # Importing manual_cast function

# Define the number of channels for different layers in the SD_V12 and SD_XL models
SD_V12_CHANNELS = [320] * 4 + [640] * 4 + [1280] * 4 + [1280] * 6 + [640] * 6 + [320] * 6 + [1280] * 2
SD_XL_CHANNELS = [640] * 8 + [1280] * 40 + [1280] * 60 + [640] * 12 + [1280] * 20

# Function to perform scaled dot-product attention with optimizations
def sdp(q, k, v, extra_options):
    return attention.optimized_attention(q, k, v, heads=extra_options["n_heads"], mask=None)

# ImageProjModel class to project image embeddings to the desired dimension
class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens,
                                                              self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

# To_KV class to convert input to key and value for the transformer
class To_KV(torch.nn.Module):
    def __init__(self, cross_attention_dim):
        super().__init__()

        channels = SD_XL_CHANNELS if cross_attention_dim == 2048 else SD_V12_CHANNELS
        self.to_kvs = torch.nn.ModuleList(
            [torch.nn.Linear(cross_attention_dim, channel, bias=False) for channel in channels])

    def load_state_dict_ordered(self, sd):
        state_dict = []
        for i in range(4096):
            for k in ['k', 'v']:
                key = f'{i}.to_{k}_ip.weight'
                if key in sd:
                    state_dict.append(sd[key])
        for i, v in enumerate(state_dict):
            self.to_kvs[i].weight = torch.nn.Parameter(v, requires_grad=False)

# IPAdapterModel class to adapt image embeddings to the transformer
class IPAdapterModel(torch.nn.Module):
    def __init__(self, state_dict, plus, cross_attention_dim=768, clip_embeddings_dim=1024, clip_extra_context_tokens=4,
                 sdxl_plus=False):
        super().__init__()
        self.plus = plus
        if self.plus:
            self.image_proj_model = Resampler(
                dim=1280 if sdxl_plus else cross_attention_dim,
                depth=4,
              ```

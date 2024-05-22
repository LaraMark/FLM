import functools
from typing import Callable, Iterable, Union

import torch
from einops import rearrange, repeat

import ldm_patched.modules.ops
ops = ldm_patched.modules.ops.disable_weight_init

from .diffusionmodules.model import (
    AttnBlock,
    Decoder,
    ResnetBlock,
)
from .diffusionmodules.openaimodel import ResBlock, timestep_embedding
from .attention import BasicTransformerBlock

def partialclass(cls, *args, **kwargs):
    """Create a partial class with the given arguments and keyword arguments."""
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


class VideoResBlock(ResnetBlock):
    """A variant of ResnetBlock that includes a time-based stack of ResBlocks."""
    def __init__(
            self,
            out_channels,
            *args,
            dropout=0.0,
            video_kernel_size=3,
            alpha=0.0,
            merge_strategy="learned",
            **kwargs,
    ):
        """
        Initialize the VideoResBlock.

        Args:
            out_channels (int): The number of output channels.
            *args: Variable length argument list.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            video_kernel_size (int or list, optional): The kernel size for the time-based stack of ResBlocks. Defaults to 3.
            alpha (float, optional): The alpha value for the merge strategy. Defaults to 0.0.
            merge_strategy (str, optional): The merge strategy for the time-based stack of ResBlocks. Defaults to "learned".
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(out_channels=out_channels, dropout=dropout, *args, **kwargs)
        if video_kernel_size is None:
            video_kernel_size = [3, 1, 1]
        self.time_stack = ResBlock(
            channels=out_channels,
            emb_channels=0,
            dropout=dropout,
            dims=3,
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            use_checkpoint=False,
            skip_t_emb=True,
        )

        self.merge_strategy = merge_strategy
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned":
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, bs):
        """
        Get the alpha value for the merge strategy.

        Args:
            bs (int): The batch size.

        Returns:
            torch.Tensor: The alpha value.
        """
        if self.merge_strategy == "fixed":
            return self.mix_factor
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError()

    def forward(self, x, temb, skip_video=False, timesteps=None):
        """
        Forward pass of the VideoResBlock.

        Args:
            x (torch.Tensor): The input tensor.
            temb (torch.Tensor): The timestep embedding tensor.
            skip_video (bool, optional): Whether to skip the video processing. Defaults to False.
            timesteps (int, optional): The number of timesteps. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        b, c, h, w = x.shape
        if timesteps is None:
            timesteps = b

        x = super().forward(x, temb)

        if not skip_video:
            x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            x = self.time_stack(x, temb)

            alpha = self.get_alpha(bs=b // timesteps).to(x.device)
            x = alpha * x + (1.0 - alpha) * x_mix

            x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class AE3DConv(ops.Conv2d):
    """A 3D convolution wrapper for 2D convolutions."""
    def __init__(self, in_channels, out_channels, video_kernel_size=3, *args, **kwargs):
        """
        Initialize the AE3DConv.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            video_kernel_size (int or list, optional): The kernel size for the 3D convolution. Defaults to 3.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(in_channels, out_channels, *args, **kwargs)
        if isinstance(video_kernel_size, Iterable):
            padding = [int(k // 2) for k in video_kernel_size]
        else:
            padding = int(video_kernel_size // 2)

        self.time_mix_conv = ops.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=video_kernel_size,
            padding=padding,
        )

    def forward(self, input, timesteps=None, skip_video=False):
        """
        Forward pass of the AE3DConv.

        Args:
            input (torch.Tensor): The input tensor.
            timesteps (int, optional): The number of timesteps. Defaults to None.
            skip_video (bool, optional): Whether to skip the video processing. Defaults to False.

        Returns:
            torch.Tensor: The output tensor.
        """
        if timesteps is None:
            timesteps = input.shape[0]
        x = super().forward(input)
        if skip_video:
            return x
        x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
        x = self.time_mix_conv(x)
        return rearrange(x, "b c t h w -> (b t) c h w")


class AttnVideoBlock(AttnBlock):
    """A variant of AttnBlock that includes a time-based attention block."""
    def __init__(
            self, in_channels: int, alpha: float = 0, merge_strategy: str = "learned"
    ):
        """
        Initialize the AttnVideoBlock.

        Args:
            in_channels (int): The number of input channels.
            alpha (float, optional): The alpha value for the merge strategy. Defaults to 0.
            merge_strategy (str, optional): The merge strategy for the time-based attention block. Defaults to "learned".
        """
        super().__init__(in_channels)
        # no context, single headed, as in base class
        self.time_mix_block = BasicTransformerBlock(
            dim=in_channels,
            n_heads=1,
            d_head=in_channels,
            checkpoint=False,
            ff_in=True,
        )

        time_embed_dim = self.in_channels * 4
        self.video_time_embed = torch.nn.Sequential(
            ops.Linear(self.in_channels, time_embed_dim),
            torch.nn.SiLU(),
            ops.Linear(time_embed_dim, self.in_channels),
        )

        self.merge_strategy = merge_strategy
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned":
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def forward(self, x, timesteps=None, skip_time_block=False):
        """
        Forward pass of the AttnVideoBlock.

        Args:
            x (torch.Tensor): The input tensor.
            timesteps (int, optional): The number of timesteps. Defaults to None.
            skip_time_block (bool, optional): Whether to skip the time-based attention block. Defaults to False.

        Returns:
            torch.Tensor: The output tensor.
        """
        if skip_time_block:
            return super().forward(x)

        if timesteps is None:
            timesteps = x.shape[0]

        x_in = x
        x = self.attention(x)
        h, w = x.shape[2:]
        x = rearrange(x, "b c h w -> b (h w) c")

        x_mix = x
        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False)
        emb = self.video_time_embed(t_emb)  # b, n_channels
        emb = emb[:, None, :]
        x_mix = x_mix + emb

        alpha = self.get_alpha().to(x.device)
        x_mix = self.time_mix_block(x_mix, timesteps=timesteps)
        x = alpha * x + (1.0 - alpha) * x_mix  # alpha merge

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)

        return x_in + x

    def get_alpha(
            self,
    ):
        """
        Get the alpha value for the merge strategy.

        Returns:
            torch.Tensor: The alpha value.
        """
        if self.merge_strategy == "fixed":
            return self.mix_factor
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError(f"unknown merge strategy {self.merge_strategy}")



def make_time_attn(
        in_channels,
        attn_type="vanilla",
        attn_kwargs=None,
        alpha: float = 0,
        merge_strategy: str = "learned",
):
    """
    Create a time-based attention block.

    Args:
        in_channels (int): The number of input channels.
        attn_type (str, optional): The type of attention block. Defaults to "vanilla".
        attn_kwargs (dict, optional): The keyword arguments for the attention block. Defaults to None.
        alpha (float, optional): The alpha value for the merge strategy. Defaults to 0.
        merge_strategy (str, optional): The merge strategy for the time-based attention block. Defaults to "learned".

    Returns:
        Callable: The time-based attention block.
    """
    return partialclass(
        AttnVideoBlock, in_channels, alpha=alpha, merge_strategy=merge_strategy
    )


class Conv2DWrapper(torch.nn.Conv2d):
    """A wrapper for 2D convolutions with a custom forward method."""
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the Conv2DWrapper.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return super().forward(input)


class VideoDecoder(Decoder):
    """A variant of Decoder that includes time-based processing."""
    available_time_modes = ["all", "conv-only", "attn-only"]

    def __init__(
            self,
            *args,
            video_kernel_size: Union[int, list] = 3,
            alpha: float = 0.0,
            merge_strategy: str = "learned",
            time_mode: str = "conv-only",
            **kwargs,
    ):
        """
        Initialize the VideoDecoder.

        Args:
            *args: Variable length argument list.
            video_kernel_size (int or list, optional): The kernel size for the time-based convolutions. Defaults to 3.
            alpha (float, optional): The alpha value for the merge strategy. Defaults to 0.0.
            merge_strategy (str, optional): The merge strategy for the time-based processing. Defaults to "learned".
            time_mode (str, optional): The time mode for the decoder. Defaults to "conv-only".
            **kwargs: Arbitrary keyword arguments.
        """
        self.video_kernel_size = video_kernel_size
        self.alpha = alpha
        self.merge_strategy = merge_strategy
        self.

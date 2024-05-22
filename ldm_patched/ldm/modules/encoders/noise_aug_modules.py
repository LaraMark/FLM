from ..diffusionmodules.upscaling import ImageConcatWithNoiseAugmentation
from ..diffusionmodules.openaimodel import Timestep
import torch

class CLIPEmbeddingNoiseAugmentation(ImageConcatWithNoiseAugmentation):
    """
    This class extends ImageConcatWithNoiseAugmentation and introduces
    normalization based on CLIP statistics. It also includes a time-step
    embedding layer.
    """
    def __init__(self, *args, clip_stats_path=None, timestep_dim=256, **kwargs):
        """
        Initializes the class with the given arguments and keyword arguments.
        If clip_stats_path is not provided, the mean and standard deviation
        are initialized as zeros and ones respectively. Otherwise, they are
        loaded from the provided path.

        :param args: Variable length argument list
        :param clip_stats_path: Path to load CLIP mean and standard deviation
        :param timestep_dim: Dimension for time-step embedding
        :param kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)
        if clip_stats_path is None:
            clip_mean, clip_std = torch.zeros(timestep_dim), torch.ones(timestep_dim)
        else:
            clip_mean, clip_std = torch.load(clip_stats_path, map_location="cpu")
        self.register_buffer("data_mean", clip_mean[None, :], persistent=False)
        self.register_buffer("data_std", clip_std[None, :], persistent=False)
        self.time_embed = Timestep(timestep_dim)

    def scale(self, x):
        """
        Renormalizes the input tensor to have a mean of 0 and a standard deviation of 1,
        based on the loaded CLIP statistics.

        :param x: Input tensor
        :return: Renormalized tensor
        """
        x = (x - self.data_mean.to(x.device)) * 1. / self.data_std.to(x.device)
        return x

    def unscale(self, x):
        """
        Undoes the normalization applied by the scale method.

        :param x: Input tensor
        :return: Denormalized tensor
        """
        x = (x * self.data_std.to(x.device)) + self.data_mean.to(x.device)
        return x

    def forward(self, x, noise_level=None):
        """
        Performs forward pass through the model. If noise_level is not provided,
        it is randomly sampled. The input tensor is then scaled, noise is added,
        and the tensor is unscaled. The noise_level is embedded using the time-step
        embedding layer.

        :param x: Input tensor
        :param noise_level: Noise level to use; if None, a random level is sampled
        :return: Augmented tensor and embedded noise_level
        """
        if noise_level is None:
            noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        else:
            assert isinstance(noise_level, torch.Tensor)
        x = self.scale(x)
        z = self.q_sample(x, noise_level)
        z = self.unscale(z)
        noise_level = self.time_embed(noise_level)
        return z, noise_level

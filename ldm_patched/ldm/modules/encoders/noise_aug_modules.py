from ..diffusionmodules.upscaling import ImageConcatWithNoiseAugmentation
from ..diffusionmodules.openaimodel import Timestep
import torch

class CLIPEmbeddingNoiseAugmentation(ImageConcatWithNoiseAugmentation):
    """
    This class extends ImageConcatWithNoiseAugmentation and implements the CLIP embedding noise augmentation technique.
    It includes methods for scaling and unscaling the input data, and a forward method to generate the noisy input.

    :param args: Variable length argument list for ImageConcatWithNoiseAugmentation constructor
    :param clip_stats_path: Path to the saved CLIP statistics (mean and std) or None to use default values
    :param timestep_dim: Dimension of the timestep
    :param kwargs: Arbitrary keyword arguments for ImageConcatWithNoiseAugmentation constructor
    """
    def __init__(self, *args, clip_stats_path=None, timestep_dim=256, **kwargs):
        super().__init__(*args, **kwargs)
        if clip_stats_path is None:
            clip_mean, clip_std = torch.zeros(timestep_dim), torch.ones(timestep_dim)
        else:
            clip_mean, clip_std = torch.load(clip_stats_path, map_location="cpu")
        
        # Register buffers for data mean and standard deviation
        self.register_buffer("data_mean", clip_mean[None, :], persistent=False)
        self.register_buffer("data_std", clip_std[None, :], persistent=False)
        
        # Initialize the time embedding object
        self.time_embed = Timestep(timestep_dim)

    def scale(self, x):
        """
        Scale the input data to have a centered mean and unit variance.

        :param x: Input data tensor
        :return: Scaled input data tensor
        """
        x = (x - self.data_mean.to(x.device)) * 1. / self.data_std.to(x.device)
        return x

    def unscale(self, x):
        """
        Unscale the input data back to its original data stats.

        :param x: Unscaled input data tensor
        :return: Scaled input data tensor
        """
        x = (x * self.data_std.to(x.device)) + self.data_mean.to(x.device)
        return x

    def forward(self, x, noise_level=None, seed=None):
        """
        Generate noisy input data based on the input data, noise level, and seed.

        :param x: Input data tensor
        :param noise_level: Noise level tensor or None to randomly generate noise levels
        :param seed: Seed for random number generation or None to use a random seed
        :return: Noisy input data tensor and noise level tensor
        """
        if noise_level is None:
            noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        else:
            assert isinstance(noise_level, torch.Tensor)
        
        # Scale the input data
        x = self.scale(x)
        
        # Generate the noisy input data
        z = self.q_sample(x, noise_level, seed=seed)
        
        # Unscale the noisy input data
        z = self.unscale(z)
        
        # Embed the noise level
        noise_level = self.time_embed(noise_level)
        
        return z, noise_level

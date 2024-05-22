import torch


# Alias for torch.Tensor
Tensor = torch.Tensor
# Alias for torch.DeviceObjType
Device = torch.DeviceObjType
# Alias for torch.Type
Dtype = torch.Type
# Alias for torch.nn.functional.pad
pad = torch.nn.functional.pad


def _compute_zero_padding(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    """
    Computes the zero padding required for a given kernel size.

    Args:
    kernel_size (tuple[int, int] | int): Kernel size for convolution or padding.

    Returns:
    tuple[int, int]: A tuple of two integers representing the padding in the y and x dimensions.
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2


def _unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    """
    Unpacks a 2D kernel size tuple or integer into two separate integers.

    Args:
    kernel_size (tuple[int, int] | int): Kernel size for convolution or padding.

    Returns:
    tuple[int, int]: A tuple of two integers representing the kernel size in the y and x dimensions.
    """
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        assert len(kernel_size) == 2, '2D Kernel size should have a length of 2.'
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)
    return ky, kx


def gaussian(
    window_size: int, sigma: Tensor | float, *, device: Device | None = None, dtype: Dtype | None = None
) -> Tensor:
    """
    Computes the Gaussian kernel for a given window size and standard deviation.

    Args:
    window_size (int): The size of the window for the Gaussian kernel.
    sigma (Tensor | float): The standard deviation for the Gaussian distribution.
    device (Device | None, optional): The device on which to create the tensor. Defaults to None.
    dtype (Dtype | None, optional): The data type for the tensor. Defaults to None.

    Returns:
    Tensor: A 1D tensor representing the Gaussian kernel.
    """
    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def get_gaussian_kernel1d(
    kernel_size: int,
    sigma: float | Tensor,
    force_even: bool = False,
    *,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> Tensor:
    """
    Returns a 1D Gaussian kernel with a given kernel size and standard deviation.

    Args:
    kernel_size (int): The size of the Gaussian kernel.
    sigma (float | Tensor): The standard deviation for the Gaussian distribution.
    force_even (bool, optional): Whether to force the kernel size to be even. Defaults to False.
    device (Device | None, optional): The device on which to create the tensor. Defaults to None.
    dtype (Dtype | None, optional): The data type for the tensor. Defaults to None.

    Returns:
    Tensor: A 1D tensor representing the Gaussian kernel.
    """
    return gaussian(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_kernel2d(
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | Tensor,
    force_even: bool = False,
    *,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> Tensor:
    """
    Returns a 2D Gaussian kernel with a given kernel size and standard deviation.

    Args:
    kernel_size (tuple[int, int] | int): The size of the Gaussian kernel as a tuple or integer.
    sigma (tuple[float, float] | Tensor): The standard deviation for the Gaussian distribution as a tuple or tensor.
    force_even (bool, optional): Whether to force the kernel size to be even. Defaults to False.
    device (Device | None, optional): The device on which to create the tensor. Defaults to None.
    dtype (Dtype | None, optional): The data type for the tensor. Defaults to None.

    Returns:
    Tensor: A 2D tensor representing the Gaussian kernel.
    """
    sigma = torch.Tensor([[sigma, sigma]]).to(device=device, dtype=dtype)

    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even, device=device, dtype=dtype)[..., None]
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even, device=device, dtype=dtype)[..., None]

    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def _bilateral_blur(
    input: Tensor,
    guidance: Tensor | None,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
) -> Tensor:
    """
    Performs bilateral blur on an input tensor with optional guidance.

    Args:
    input (Tensor): The input tensor to apply the bilateral blur.
    guidance (Tensor | None, optional): The guidance tensor for the bilateral blur. Defaults to None.
    kernel_size (tuple[int, int] | int): The size of the blur kernel.
    sigma_color (float | Tensor): The standard deviation for color in the bilateral blur.
    sigma_space (tuple[float, float] | Tensor): The standard deviation for space in the bilateral blur.
    border_type (str, optional): The border type for padding. Defaults to 'reflect'.
    color_distance_type (str, optional): The type of color distance to use. Defaults to 'l1'.

    Returns:
    Tensor: The blurred output tensor.
    """
    pass


def bilateral_blur(
    input: Tensor,
    kernel_size: tuple[int, int] | int = (13, 13),
    sigma_color: float | Tensor = 3.0,
    sigma_space: tuple[float, float] | Tensor = 3.0,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
) -> Tensor:
    """
    Applies bilateral blur to an input tensor.

    Args:
    input (Tensor): The input tensor to apply the bilateral blur.
    kernel_size (tuple[int, int] | int, optional): The size of the blur kernel. Defaults to (13, 13).
    sigma_color (float | Tensor, optional): The standard deviation for color in the bilateral blur. Defaults to 3.0.
    sigma_space (tuple[float, float] | Tensor, optional): The standard deviation for space in the bilateral blur. Defaults to (3.0, 3.0).
    border_type (str, optional): The border type for padding. Defaults to 'reflect'.
    color_distance_type (str, optional): The type of color distance to use. Defaults to 'l1'.

    Returns:
    Tensor: The blurred output tensor.
    """
    return _bilateral_blur(input, None, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


def adaptive_anisotropic_filter(x, g=None):
    """
    Applies adaptive anisotropic filtering to an input tensor.

    Args:
    x (Tensor): The input tensor to apply the filter.
    g (Tensor | None, optional): The guidance tensor for the filter. Defaults to None.

    Returns:
    Tensor: The filtered output tensor.
    """
    pass


def joint_bilateral_blur(
    input: Tensor,
    guidance: Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
) -> Tensor:
    """
    Applies joint bilateral blur to an input tensor with guidance.

    Args:
    input (Tensor): The input tensor to apply the bilateral blur.
    guidance (Tensor): The guidance tensor for the bilateral blur.
    kernel_size (tuple[int, int] | int): The size of the blur kernel.
    sigma_color (float | Tensor): The standard deviation for color in the bilateral blur.
    sigma_space (tuple[float, float] | Tensor): The standard deviation for space in the bilateral blur.
    border_type (str, optional): The border type for padding. Defaults to 'reflect'.
    color_distance_type (str, optional): The type of color distance to use. Defaults to 'l1'.

    Returns:
    Tensor: The blurred output tensor.
    """
    return _bilateral_blur(input, guidance, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


class _BilateralBlur(torch.nn.Module):
    """
    Base class for bilateral blur operations.
    """
    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        sigma_color: float | Tensor,
        sigma_space: tuple[float, float] | Tensor,
        border_type: str = 'reflect',
        color_distance_type: str = "l1",
    ) -> None:
        """
        Initializes the base class for bilateral blur operations.

        Args:
        kernel_size (tuple[int, int] | int): The size of the blur kernel.
        sigma_color (float | Tensor): The standard deviation for color in the bilateral blur.
        sigma_space (tuple[float, float] | Tensor): The standard deviation for space in the bilateral blur.
        border_type (str, optional): The border type for padding. Defaults to 'reflect'.
        color_distance_type (str, optional): The type of color distance to use. Defaults to 'l1'.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.border_type = border_type
        self.color_distance_type = color_distance_type

    def __repr__(self) -> str:
        """
        Returns a string representation of the class.

        Returns:
        str: The string representation of the class.
        """
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"sigma_color={self.sigma_color}, "
            f"sigma_space={self.sigma_space}, "
            f"border_type={self.border_type}, "
            f"color_distance_

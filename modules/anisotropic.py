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
    kernel_size: A tuple of ints or an int specifying the kernel size.

    Returns:
    A tuple of ints representing the padding required in y and x dimensions.
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2


def _unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    """
    Unpacks a 2D kernel size tuple or an int representing a square kernel size.

    Args:
    kernel_size: A tuple of ints or an int specifying the kernel size.

    Returns:
    A tuple of ints representing the kernel size in y and x dimensions.
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
    Computes the Gaussian kernel for a given window size and sigma.

    Args:
    window_size: An int specifying the window size.
    sigma: A float or a Tensor of shape (batch_size,) specifying the standard deviation.
    device: A Device or None to specify the device where the tensor should be allocated.
    dtype: A Dtype or None to specify the data type of the tensor.

    Returns:
    A 1D Gaussian kernel as a Tensor.
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
    Returns a 1D Gaussian kernel with a given kernel size and sigma.

    Args:
    kernel_size: An int specifying the kernel size.
    sigma: A float or a Tensor of shape (batch_size,) specifying the standard deviation.
    force_even: A bool indicating whether to force the kernel size to be even.
    device: A Device or None to specify the device where the tensor should be allocated.
    dtype: A Dtype or None to specify the data type of the tensor.

    Returns:
    A 1D Gaussian kernel as a Tensor.
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
    Returns a 2D Gaussian kernel with a given kernel size and sigma.

    Args:
    kernel_size: A tuple of ints or an int specifying the kernel size.
    sigma: A tuple of floats or a Tensor of shape (batch_size, 2) specifying the standard deviation.
    force_even: A bool indicating whether to force the kernel size to be even.
    device: A Device or None to specify the device where the tensor should be allocated.
    dtype: A Dtype or None to specify the data type of the tensor.

    Returns:
    A 2D Gaussian kernel as a Tensor.
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
    Performs bilateral blur on the input tensor with guidance.

    Args:
    input: A Tensor to be blurred.
    guidance: A Tensor for guidance or None if not provided.
    kernel_size: A tuple of ints or an int specifying the kernel size.
    sigma_color: A float or a Tensor of shape (batch_size,) specifying the color sigma.
    sigma_space: A tuple of floats or a Tensor of shape (batch_size, 2) specifying the space sigma.
    border_type: A str specifying the border type for padding.
    color_distance_type: A str specifying the color distance type ('l1' or 'l2').

    Returns:
    A blurred Tensor.
    """
    # ... (rest of the function)


def bilateral_blur(
    input: Tensor,
    kernel_size: tuple[int, int] | int = (13, 13),
    sigma_color: float | Tensor = 3.0,
    sigma_space: tuple[float, float] | Tensor = 3.0,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
) -> Tensor:
    """
    Performs bilateral blur on the input tensor.

    Args:
    input: A Tensor to be blurred.
    kernel_size: A tuple of ints or an int specifying the kernel size.
    sigma_color: A float or a Tensor of shape (batch_size,) specifying the color sigma.
    sigma_space: A tuple of floats or a Tensor of shape (batch_size, 2) specifying the space sigma.
    border_type: A str specifying the border type for padding.
    color_distance_type: A str specifying the color distance type ('l1' or 'l2').

    Returns:
    A blurred Tensor.
    """
    return _bilateral_blur(input, None, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


def adaptive_anisotropic_filter(x, g=None):
    """
    Performs adaptive anisotropic filtering on the input tensor.

    Args:
    x: A Tensor to be filtered.
    g: A guidance Tensor or None if not provided.

    Returns:
    A filtered Tensor.
    """
    # ... (rest of the function)


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
    Performs joint bilateral blur on the input tensor with guidance.

    Args:
    input: A Tensor to be blurred.
    guidance: A Tensor for guidance.
    kernel_size: A tuple of ints or an int specifying the kernel size.
    sigma_color: A float or a Tensor of shape (batch_size,) specifying the color sigma.
    sigma_space: A tuple of floats or a Tensor of shape (batch_size, 2) specifying the space sigma.
    border_type: A str specifying the border type for padding.
    color_distance_type: A str specifying the color distance type ('l1' or 'l2').

    Returns:
    A blurred Tensor.
    """
    return _bilateral_blur(input, guidance, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


class _BilateralBlur(torch.nn.Module):
    """
    Base class for BilateralBlur and JointBilateralBlur.
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
        Initializes the _BilateralBlur base class.

        Args:
        kernel_size: A tuple of ints or an int specifying the kernel size.
        sigma_color: A float or a Tensor of shape (batch_size,) specifying the color sigma.
        sigma_space: A tuple of floats or a Tensor of shape (batch_size, 2) specifying the space sigma.
        border_type: A str specifying the border type for padding.
        color_distance_type: A str specifying the color distance type ('l1' or 'l2').
        """
        # ... (rest of the class)


class BilateralBlur(_BilateralBlur):
    """
    Applies bilateral blur to the input tensor.
    """
    def forward(self, input: Tensor) -> Tensor:
        """
        Applies bilateral blur to the input tensor.

        Args:
        input: A Tensor to be blurred.

        Returns:
        A blurred Tensor.
        """
        # ... (rest of the class)


class JointBilateralBlur(_BilateralBlur):
    """
    Applies joint bilateral blur to the input tensor with guidance.
    """
    def forward(self, input: Tensor, guidance: Tensor) -> Tensor:
        """
        Applies joint bilateral blur to the input tensor with guidance.

        Args:
        input: A Tensor to be blurred.
        guidance: A Tensor for guidance.

        Returns:
        A blurred Tensor.
        """
        # ... (rest of the class)

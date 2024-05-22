import cv2
import numpy as np
import modules.advanced_parameters as advanced_parameters  # Import the advanced_parameters module


def centered_canny(x: np.ndarray):
    """
    Applies the Canny edge detector to a grayscale image and returns a centered canny edge map.

    :param x: Grayscale image (np.ndarray) of shape (height, width)
    :return: Centered canny edge map (np.ndarray) of shape (height, width)
    """
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2 and x.dtype == np.uint8

    y = cv2.Canny(x, int(advanced_parameters.canny_low_threshold), int(advanced_parameters.canny_high_threshold))
    y = y.astype(np.float32) / 255.0
    return y


def centered_canny_color(x: np.ndarray):
    """
    Applies the centered_canny function to each color channel of a color image and returns a centered canny edge map for each channel.

    :param x: Color image (np.ndarray) of shape (height, width, 3)
    :return: Color centered canny edge map (np.ndarray) of shape (height, width, 3)
    """
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3 and x.shape[2] == 3

    result = [centered_canny(x[..., i]) for i in range(3)]
    result = np.stack(result, axis=2)
    return result


def pyramid_canny_color(x: np.ndarray):
    """
    Applies the centered_canny_color function to a color image at different scales and combines the results to create a final edge map.

    :param x: Color image (np.ndarray) of shape (height, width, 3)
    :return: Edge map (np.ndarray) of shape (height, width)
   

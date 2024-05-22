import cv2
import numpy as np
import torch


def compute_increased_bbox(bbox, increase_area, preserve_aspect=True):
    """
    Compute a new bounding box with increased area while preserving the aspect ratio if desired.

    Parameters:
    bbox (tuple): Original bounding box as (left, top, right, bottom).
    increase_area (float): The factor to increase the area of the bounding box.
    preserve_aspect (bool): Whether to preserve the aspect ratio of the bounding box.

    Returns:
    tuple: New bounding box as (left, top, right, bottom).
    """
    left, top, right, bot = bbox
    width = right - left
    height = bot - top

    if preserve_aspect:
        width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
        height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))
    else:
        width_increase = height_increase = increase_area

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    return (left, top, right, bot)


def get_valid_bboxes(bboxes, h, w):
    """
    Get the valid bounding boxes within the image boundaries.

    Parameters:
    bboxes (tuple): Bounding boxes as (left, top, right, bottom).
    h (int): Image height.
    w (int): Image width.

    Returns:
    tuple: Valid bounding boxes as (left, top, right, bottom).
    """
    left = max(bboxes[0], 0)
    top = max(bboxes[1], 0)
    right = min(bboxes[2], w)
    bottom = min(bboxes[3], h)

    return (left, top, right, bottom)


def align_crop_face_landmarks(img,
                              landmarks,
                              output_size,
                              transform_size=None,
                              enable_padding=True,
                              return_inverse_affine=False,
                              shrink_ratio=(1, 1)):
    """
    Align and crop face using landmarks.

    Parameters:
    img (numpy.ndarray): Input image.
    landmarks (numpy.ndarray): Landmarks of the face.
    output_size (int): Output face size.
    transform_size (int): Transform size.
    enable_padding (bool): Enable padding.
    return_inverse_affine (bool): Return inverse affine matrix.
    shrink_ratio (tuple): Shrink ratio for height and width.

    Returns:
    numpy.ndarray: Cropped face.
    """
    pass  # Implementation not shown in the provided code.


def paste_face_back(img, face, inverse_affine):
    """


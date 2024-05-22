import numpy as np
import datetime
import random
import math
import os
import cv2

# Use LANCZOS resampling method for Image.resize() if it's available, otherwise default to LANCZOS
LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

def erode_or_dilate(x, k):
    """
    Perform image dilation or erosion based on the input kernel value 'k'.
    :param x: The input image as a numpy array.
    :param k: The kernel size for dilation or erosion. Positive values for dilation, negative for erosion.
    :return: The processed image as a numpy array.
    """
    k = int(k)
    if k > 0:
        return cv2.dilate(x, kernel=np.ones(shape=(3, 3), dtype=np.uint8), iterations=k)
    if k < 0:
        return cv2.erode(x, kernel=np.ones(shape=(3, 3), dtype=np.uint8), iterations=-k)
    return x

def resample_image(im, width, height):
    """
    Resample an image using the LANCZOS method.
    :param im: The input image as a numpy array.
    :param width: The desired width of the output image.
    :param height: The desired height of the output image.
    :return: The resampled image as a numpy array.
    """
    im = Image.fromarray(im)
    im = im.resize((int(width), int(height)), resample=LANCZOS)
    return np.array(im)

def resize_image(im, width, height, resize_mode=1):
    """
    Resize an image based on the specified resize_mode, width, and height.
    :param resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        :param im: The input image as a numpy array.
        :param width: The desired width of the output image.
        :param height: The desired height of the output image.
    :return: The resized image as a numpy array.
    """
    im = Image.fromarray(im)

    def resize(im, w, h):
        return im.resize((w, h), resample=LANCZOS)

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return np.array(res)

def get_shape_ceil(h, w):
    """
    Calculate the nearest multiple of 64 greater than or equal to the product of h and w.
    :param h: The height value.
    :param w: The width value.
    :return: The nearest multiple of 64.
    """
    return math.ceil(((h * w) ** 0.5) / 64.0) * 64.0

def get_image_shape_ceil(im):
    """
    Calculate the nearest multiple of 64 greater than or equal to the product of the image's height and width.
    :param im: The input image as a numpy array.
    :return: The nearest multiple of 64.
    """
    H, W = im.shape[:2]
    return get_shape_ceil(H, W)

def set_image_shape_ceil(im, shape_ceil):
    """
    Resize the input image to the nearest multiple of 64 greater than or equal to the product of the image's height and width.
    :param im: The input image as a numpy array.
    :param shape_ceil: The nearest multiple of 64.
    :return: The resized image as a numpy array.
    """
    shape_ceil = float(shape_ceil)

    H_origin, W_origin, _ = im.shape
    H, W = H_origin, W_origin

    for _ in range(256):
        current_shape_ceil = get_shape_ceil(H, W)
        if abs(current_shape_ceil - shape_ceil) < 0.1:
            break
        k = shape_ceil / current_shape_ceil
        H = int(round(float(H) * k / 64.0) * 64)
        W = int(round(float(W) * k / 64.0) * 64)

    if H == H_origin and W == W_origin:
        return im

    return resample_image(im, width=W, height=H)

def HWC3(x):
    """
    Convert the input image to a 3-channel format (RGB) if it's not already.
    :param x: The input image as a numpy array.
    :return: The 3-channel image as a numpy array.
    """
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def remove_empty_str(items, default=None):
    """
    Remove empty strings from the input list and return the updated list.
    :param items: The input list.
    :param default: The default value to return if the list is empty.
    :return: The updated list or the default value.
    """
    items = [x for x in items if x != ""]
    if len(items) == 0 and default is not None:
        return [default]
    return items

def join_prompts(*args, **kwargs):
    """
    Join the input strings into a single string with commas as separators.
    :param args: The input strings.
    :param kwargs: Additional keyword arguments.
    :return: The joined string.
    """
    prompts = [str(x) for x in args if str(x) != ""]
    if len(prompts) == 0:
        return ""
    if len(prompts) == 1:
        return prompts[0]
    return ', '.join(prompts)

def generate_temp_filename(folder='./outputs/', extension='png'):
    """
    Generate a temporary filename for storing the output image.
    :param folder: The output folder path.
    :param extension: The file extension.
    :return: The date string, the absolute path of the temporary file, and the filename.
    """
    current_time = datetime.datetime.now()
    date_string = current_time.strftime("%Y-%m-%d")
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    random_number = random.randint(1000, 9999)
    filename = f"{time_string}_{random_number}.{extension}"
    result = os.path.join(folder, date_string, filename)
    return date_string, os.path.abspath(os.path.realpath(result)), filename

def get_files_from_folder(folder_path, exensions=None, name_filter=None):
    """
    Get a list of files from the input folder with the specified file extensions and name filter.
    :param folder_path: The folder path.
    :param exensions: The allowed file extensions.
    :param name_filter: The name filter for the files.
    :return: The sorted list of file paths.
    """
    if not os.path.isdir(folder_path):
        raise ValueError("Folder path is not a valid directory.")

    filenames = []

    for root, dirs, files in os.walk(folder_path):
        relative_path = os.path.relpath(root, folder_path)
        if relative_path == ".":
            relative_path = ""
        for filename in files:
            _, file_extension = os.path.splitext(filename)
            if (exensions == None or file_extension.lower() in exensions) and (name_filter == None or name_filter in _):
                path = os.path.join(relative_path, filename)
                filenames.append(path)

    return sorted(filenames, key=lambda x: -1 if os.sep in x else 1)

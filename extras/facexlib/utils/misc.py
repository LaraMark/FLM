import cv2
import os
import os.path as osp
import torch
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse

# ROOT_DIR is set to the parent directory of the current file's grandparent directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """
    Write image to file.

    This function writes an image array to a specified file path. If the parent folder of the file path does not exist,
    it can automatically create the folder.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Additional parameters for OpenCV's imwrite function.
        auto_mkdir (bool): If set to True, automatically create the parent folder if it doesn't exist.

    Returns:
        bool: Returns True if the image is successfully written, False otherwise.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """
    Numpy array to tensor.

    This function converts a numpy array or a list of numpy arrays to PyTorch tensors. It also handles BGR to RGB conversion
    and float32 data type conversion.

    Args:
        imgs (list[ndarray] | ndarray): Input images as a list or a single numpy array.
        bgr2rgb (bool): If set to True, converts BGR to RGB format.
        float32 (bool): If set to True, converts the data type to float32.

    Returns:
        list[tensor] | tensor: A list or a single tensor, depending on the input.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def load_file_from_url(url, model_dir=None, progress=True, file_name=None, save_dir=None):
    """
    Load a file from a URL and save it to a local directory.

    This function downloads a file from a given URL and saves it to a specified directory. If the directory does not exist,
    it creates the directory.

    Args:
        url (str): The URL of the file to download.
        model_dir (str, optional): The directory where the file will be saved. Defaults to None.
        progress (bool, optional): If set to True, display the download progress. Defaults to True.
        file_name (str, optional): The name of the file to save. Defaults to None.
        save_dir (str, optional): The directory where the file will be saved. Defaults to None.

    Returns:
        str: The local file path of the downloaded file.
    """
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    if save_dir is None:
        save_dir = os.path.join(ROOT_DIR, model_dir)
    os.makedirs(save_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(save_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """
    Scan a directory and find files with a specific suffix.

    This function scans a directory and returns a generator for file paths that match a specified suffix.

    Args:
        dir_path (str): The directory to scan.
        suffix (str | tuple(str), optional): The file suffix to search for. Defaults to None.
        recursive (bool, optional): If set to True, scan subdirectories recursively. Defaults to False.
        full_path (bool, optional): If set to True, include the full path of the files. Defaults to False.

    Yields:
        str: The relative path of the file if full_path is False, or the full path if full_path is True.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

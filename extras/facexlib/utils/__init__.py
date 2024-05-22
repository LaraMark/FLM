from .face_utils import align_crop_face_landmarks, compute_increased_bbox, get_valid_bboxes, paste_face_back
from .misc import img2tensor, load_file_from_url, scandir

# The __all__ variable is a list of module-level public objects that can be imported by using the
# `from module_name import *` syntax. In this case, we are making the following functions and
# classes available for import:
# - align_crop_face_landmarks
# - compute_increased_bbox
# - get_valid_bboxes
# - load_file_from_url
# - paste_face_back
# - img2tensor
# - scandir

# The `align_crop_face_landmarks` function takes an image and a list of facial landmarks, and returns
# a cropped and aligned version of the image using the specified landmarks.
#
# Arguments:
# - image: a PIL Image object
# - landmarks: a list of (x, y) coordinates representing the facial landmarks
#
# Returns:
# - a PIL Image object representing the cropped and aligned face
def align_crop_face_landmarks(image, landmarks):
    ...

# The `compute_increased_bbox` function takes a bounding box and a scaling factor, and returns a new
# bounding box that is scaled up by the specified factor.
#
# Arguments:
# - bbox: a list of four integers representing the bounding box coordinates (x1, y1, x2, y2)
# - scale: a float representing the scaling factor
#
# Returns:
# - a list of four integers representing the increased bounding box coordinates
def compute_increased_bbox(bbox, scale):
    ...

# The `get_valid_bboxes` function takes a list of bounding boxes and filters out any boxes that are
# too small or too large, or that overlap too much with other boxes.
#
# Arguments:
# - bboxes: a list of lists of four integers representing the bounding box coordinates
# - min_size: an integer representing the minimum allowed bounding box size
# - max_size: an integer representing the maximum allowed bounding box size
# - iou_thresh: a float representing the maximum allowed intersection over union between boxes
#
# Returns:
# - a list of lists of four integers representing the valid bounding boxes
def get_valid_bboxes(bboxes, min_size, max_size, iou_thresh):
    ...

# The `load_file_from_url` function takes a URL and returns the contents of the file at that URL as
# a string.
#
# Arguments:
# - url: a string representing the URL of the file to download
#
# Returns:
# - a string containing the contents of the file
def load_file_from_url(url):
    ...

# The `paste_face_back` function takes an image, a cropped face, and a bounding box, and pastes the
# cropped face back into the original image at the specified location.
#
# Arguments:
# - image: a PIL Image object
# - cropped_face: a PIL Image object representing the cropped face
# - bbox: a list of four integers representing the bounding box coordinates
#
# Returns:
# - a PIL Image object with the cropped face pasted in
def paste_face_back(image, cropped_face, bbox):
    ...

# The `img2tensor` function takes a PIL Image object and returns a PyTorch tensor with the same
# pixel values.
#
# Arguments:
# - img: a PIL Image object
#
# Returns:
# - a PyTorch tensor with the same pixel values as the input image
def img2tensor(img):
    ...

# The `scandir` function is a context manager that yields the entries in a directory one at a time,
# in alphabetical order.
#
# Yields:
# - a DirectoryEntry object representing the next entry in the directory
def scandir(path):
    ...

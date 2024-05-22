import cv2
import numpy as np
import os
import torch
from torchvision.transforms.functional import normalize

from extras.facexlib.detection import init_detection_model  # Init face detection model
from extras.facexlib.parsing import init_parsing_model  # Init face parsing model
from extras.facexlib.utils.misc import img2tensor, imwrite  # Utility functions for image processing



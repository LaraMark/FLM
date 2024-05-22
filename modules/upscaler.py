import os
import torch
import modules.core as core
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from collections import OrderedDict
from modules.config import path_upscale_models

# Define the filename of the pre-trained upscaling model
model_filename = os.path.join(path_upscale_models, 'fooocus_upscaler_s409985e5.bin')

# Create an instance of the ImageUpscaleWithModel class
opImageUpscaleWithModel = ImageUpscaleWithModel()

# Initialize the upscaling model as a global variable
model = None

def perform_upscale(img):
    """
    Perform upscaling of the input image using the pre-trained upscaling model.

    Args:
    img (numpy.ndarray): The input image to be upscaled, in the format of a numpy array.

    Returns:
    numpy.ndarray: The upscaled image, in the format of a numpy array.
    """
    global model

    # Print a message indicating the start of the upscaling process
    print(f'Upscaling image with shape {str(img.shape)} ...')

    # Initialize the upscaling model if it hasn't been initialized yet
    if model is None:
        # Load the pre-trained model weights from the binary file
        sd = torch.load(model_filename)

        # Create a new ordered dictionary to store the model weights with modified keys
        sdo = OrderedDict()
        for k, v in sd.items():
            # Replace the key 'residual_block_' with 'RDB' for each weight
            sdo[k.replace('residual_block_', 'RDB')] = v

        # Delete the old dictionary of weights
        del sd

        # Initialize the upscaling model with the modified weights
        model = ESRGAN(sdo)

        # Move the model to the CPU and set it to evaluation mode
        model.cpu()
        model.eval()

    # Convert the input image to a PyTorch tensor
    img = core.numpy_to_pytorch(img)

    # Perform the upscaling using the ImageUpscaleWithModel class
    img = opImageUpscaleWithModel.upscale(model, img)[0]

    # Convert the upscaled image back to a numpy array
    img = core.pytorch_to_numpy(img)[0]

    # Return the upscaled image
    return img

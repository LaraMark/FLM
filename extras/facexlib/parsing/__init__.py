import torch

from extras.facexlib.utils import load_file_from_url  # Import the function to load file from URL

# Import the required models from the corresponding modules
from .bisenet import BiSeNet
from .parsenet import ParseNet

def init_parsing_model(model_name='bisenet', half=False, device='cuda', model_rootpath=None):
    r"""
    Initialize a parsing model with the given configuration.

    Args:
        model_name (str, optional): The name of the model to initialize. Defaults to 'bisenet'.
        half (bool, optional): Whether to use half-precision floating point (FP16) computation. Defaults to False.
        device (str, optional): The device where the model will be initialized. Defaults to 'cuda'.
        model_rootpath (str, optional): The root path to save the model files. Defaults to None.

    Returns:
        The initialized parsing model.

    Raises:
        NotImplementedError: If the provided model_name is not supported.
    """

    if model_name == 'bisenet':
        model = BiSeNet(num_class=19)  # Initialize BiSeNet model with 19 classes
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth'
    elif model_name == 'parsenet':
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)  # Initialize ParseNet model with 19 classes
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url,  # Download the model weights from the provided URL
        model_dir='facexlib/weights',  # Directory to save the model weights
        progress=True,  # Display download progress
        file_name=None,  # Use the original file name from the URL
        save_dir=model_rootpath)  # Save the model weights to the specified root path

    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)  # Load the model weights
    model.load_state_dict(load_net, strict=True)  # Load the state dictionary to the model
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)  # Move the model to the specified device
    return model

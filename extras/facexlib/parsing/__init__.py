import torch

from extras.facexlib.utils import load_file_from_url  # Import the function to load file from URL

# Import the required models from the corresponding modules
from .bisenet import BiSeNet
from .parsenet import ParseNet

def init_parsing_model(model_name='bisenet', half=False, device='cuda', model_rootpath=None):
    """
    Initialize the parsing model with the given configuration.

    Args:
        model_name (str): The name of the model to initialize. Either 'bisenet' or 'parsenet'. Default is 'bisenet'.
        half (bool): Whether to use half-precision floating point (FP16) computation. Default is False.
        device (str): The device where the model will be initialized. Either 'cpu' or 'cuda'. Default is 'cuda'.
        model_rootpath (str): The root path to save the model files. If not provided, the files will be saved in the
                              default directory.

    Returns:
        The initialized parsing model.

    """
    if model_name == 'bisenet':
        model = BiSeNet(num_class=19)  # Initialize BiSeNet model with 19 classes
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth'
    elif model_name == 'parsenet':
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)  # Initialize ParseNet model with 19 classes
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')  # Raise an error if the model is not supported

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)  # Load the state dictionary from the file
    model.load_state_dict(load_net, strict=True)  # Load the state dictionary to the model
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)  # Move the model to the specified device
    return model

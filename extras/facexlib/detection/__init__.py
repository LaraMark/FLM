import torch
from copy import deepcopy

from extras.facexlib.utils import load_file_from_url  # Import the function to load a file from a URL
from .retinaface import RetinaFace  # Import the RetinaFace class

def init_detection_model(model_name, half=False, device='cuda', model_rootpath=None):
    """
    Initialize a detection model with the specified name and parameters.

    Args:
    model_name (str): The name of the detection model.
    half (bool, optional): Whether to use half-precision floating point computation. Defaults to False.
    device (str, optional): The device where the model will be loaded. Defaults to 'cuda'.
    model_rootpath (str, optional): The root path for the model file. Defaults to None.

    Returns:
    model (RetinaFace): The initialized detection model.
    """
    if model_name == 'retinaface_resnet50':
        model = RetinaFace(network_name='resnet50', half=half, device=device)  # Initialize the RetinaFace model with ResNet50 backbone
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth'  # URL for the pre-trained model
    elif model_name == 'retinaface_mobile0.25':
        model = RetinaFace(network_name='mobile0.25', half=half, device=device)  # Initialize the RetinaFace model with MobileNet0.25 backbone
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth'  # URL for the pre-trained model
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')  # Raise an error if the specified model name is not implemented

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)  # Download and save the pre-trained model

    # TODO: clean pretrained model  # TODO: Remove this line once the pre-trained model is cleaned
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)  # Load the pre-trained model
    # remove unnecessary 'module.'  # Remove the 'module.' prefix from the keys of the loaded model
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=True)  # Load the state dictionary of the pre-trained model to the current model
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)  # Move the model to the specified device
    return model

import torch
from copy import deepcopy

from extras.facexlib.utils import load_file_from_url  # Importing load_file_from_url from facexlib utils
from .retinaface import RetinaFace  # Importing RetinaFace class

def init_detection_model(model_name, half=False, device='cuda', model_rootpath=None):
    """
    Initialize a detection model based on the given model name.

    :param model_name: The name of the detection model to initialize.
    :param half: A boolean indicating whether to use half-precision floating point computation.
    :param device: The device where the model will be loaded.
    :param model_rootpath: The root path for saving the model files.
    :return: The initialized detection model.
    """
    if model_name == 'retinaface_resnet50':
        model = RetinaFace(network_name='resnet50', half=half, device=device)  # Initialize RetinaFace with resnet50
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth'
    elif model_name == 'retinaface_mobile0.25':
        model = RetinaFace(network_name='mobile0.25', half=half, device=device)  # Initialize RetinaFace with mobile0.25
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')  # Raise an error if the model name is not recognized

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)

    # TODO: clean pretrained model
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)  # Load the pre-trained model

    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v  # Update the key to remove 'module.'
            load_net.pop(k)  # Remove the original key with 'module.'

    model.load_state_dict(load_net, strict=True)  # Load the state dictionary to the model
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)  # Move the model to the specified device
    return model

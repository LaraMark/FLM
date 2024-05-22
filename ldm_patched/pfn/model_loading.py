import logging as logger  # Importing the logging module for debugging purposes

from .architecture.DAT import DAT
from .architecture.face.codeformer import CodeFormer
from .architecture.face.gfpganv1_clean_arch import GFPGANv1Clean
from .architecture.face.gfpganv1_clean_arch import RestoreFormer
from .architecture.HAT import HAT
from .architecture.LaMa import LaMa
from .architecture.OmniSR.OmniSR import OmniSR
from .architecture.RRDB import RRDBNet as ESRGAN
from .architecture.SCUNet import SCUNet
from .architecture.SPSR import SPSRNet as SPSR
from .architecture.SRVGG import SRVGGNetCompact as RealESRGANv2
from .architecture.SwiftSRGAN import Generator as SwiftSRGAN
from .architecture.Swin2SR import Swin2SR
from .architecture.SwinIR import SwinIR
from .types import PyTorchModel


class UnsupportedModel(Exception):
    pass


def load_state_dict(state_dict) -> PyTorchModel:
    """
    Load a state dictionary into a PyTorch model architecture.

    This function takes a state dictionary as input and returns an instance of a PyTorch model.
    It is designed to handle different architectures by checking specific keys in the state dictionary.

    :param state_dict: A state dictionary containing the model's parameters.
    :return: A PyTorch model instance.
    """
    logger.debug(f"Loading state dict into pytorch model arch")

    # Process the state dictionary to match the desired model's parameters
    state_dict_keys = list(state_dict.keys())

    if "params_ema" in state_dict_keys:
        state_dict = state_dict["params_ema"]
    elif "params-ema" in state_dict_keys:
        state_dict = state_dict["params-ema"]
    elif "params" in state_dict_keys:
        state_dict = state_dict["params"]

    state_dict_keys = list(state_dict.keys())

    # Initialize the model based on the state dictionary keys
    if "body.0.weight" in state_dict_keys and "body.1.weight" in state_dict_keys:
        model = RealESRGANv2(state_dict)
    # ... (other model initialization cases)
    else:
        try:
            model = ESRGAN(state_dict)
        except:
            # pylint: disable=raise-missing-from
            raise UnsupportedModel  # Raise an exception if the model is unsupported

    return model

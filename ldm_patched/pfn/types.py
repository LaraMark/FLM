from typing import Union  # Importing Union from typing module

from .architecture.DAT import DAT  # Importing DAT class
from .architecture.face.codeformer import CodeFormer  # Importing CodeFormer class
from .architecture.face.gfpganv1_clean_arch import GFPGANv1Clean  # Importing GFPGANv1Clean class
from .architecture.face.restoreformer_arch import RestoreFormer  # Importing RestoreFormer class
from .architecture.HAT import HAT  # Importing HAT class
from .architecture.LaMa import LaMa  # Importing LaMa class
from .architecture.OmniSR.OmniSR import OmniSR  # Importing OmniSR class
from .architecture.RRDB import RRDBNet as ESRGAN  # Importing ESRGAN class
from .architecture.SCUNet import SCUNet  # Importing SCUNet class
from .architecture.SPSR import SPSRNet as SPSR  # Importing SPSR class
from .architecture.SRVGG import SRVGGNetCompact as RealESRGANv2  # Importing RealESRGANv2 class
from .architecture.SwiftSRGAN import Generator as SwiftSRGAN  # Importing SwiftSRGAN class
from .architecture.Swin2SR import Swin2SR  # Importing Swin2SR class
from .architecture.SwinIR import SwinIR  # Importing SwinIR class

# Define a tuple of all the super-resolution models
PyTorchSRModels = (
    RealESRGANv2,
    SPSR,
    SwiftSRGAN,
    ESRGAN,
    SwinIR,
    Swin2SR,
    HAT,
    OmniSR,
    SCUNet,
    DAT,
)

# Define a Union type for all the super-resolution models
PyTorchSRModel = Union[
    RealESRGANv2,
    SPSR,
    SwiftSRGAN,
    ESRGAN,
    SwinIR,
    Swin2SR,
    HAT,
    OmniSR,
    SCUNet,
    DAT,
]

# A function to check if a given model is a super-resolution model
def is_pytorch_sr_model(model: object):
    return isinstance(model, PyTorchSRModels)

# Define a tuple of all the face restoration models
PyTorchFaceModels = (GFPGANv1Clean, RestoreFormer, CodeFormer)

# Define a Union type for all the face restoration models
PyTorchFaceModel = Union[GFPGANv1Clean, RestoreFormer, CodeFormer]

# A function to check if a given model is a face restoration model
def is_pytorch_face_model(model: object):
    return isinstance(model, PyTorchFaceModels)

# Define a tuple of all the inpainting models
PyTorchInpaintModels = (LaMa,)

# Define a Union type for all the inpainting models
PyTorchInpaintModel = Union[LaMa]

# A function to check if a given model is an inpainting model
def is_pytorch_inpaint_model(model: object):
    return isinstance(model, PyTorchInpaintModels)

# Define a tuple of all the models
PyTorchModels = (*PyTorchSRModels, *PyTorchFaceModels, *PyTorchInpaintModels)

# Define a Union type for all the models
PyTorchModel = Union[PyTorchSRModel, PyTorchFaceModel, PyTorchInpaintModel]

# A function to check if a given model is a valid model
def is_pytorch_model(model: object):
    return isinstance(model, PyTorchModels)

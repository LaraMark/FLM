import os

import ldm_patched.modules.sd  # Importing the necessary module

def first_file(path, filenames):
    """
    This function returns the first file in the given list of filenames that exists in the given path.
    If no files exist, it returns None.
    """
    for f in filenames:
        p = os.path.join(path, f)
        if os.path.exists(p):
            return p
    return None

def load_diffusers(model_path, output_vae=True, output_clip=True, embedding_directory=None):
    """


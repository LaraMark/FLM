import ldm_patched.modules.utils
import ldm_patched.utils.path_utils
import torch

def load_hypernetwork_patch(path, strength):
    """
    Loads a hypernetwork model from the given path and applies some transformations to its components.

    Args:
    path (str): The path to the saved hypernetwork model.
    strength (float): The strength value to be applied to the hypernetwork model.

    Returns:
    A hypernetwork patch object if the hypernetwork model is loaded successfully, None otherwise.
    """
    sd = ldm_patched.modules.utils.load_torch_file(path, safe_load=True)
    # ...

class HypernetworkLoader:
    """
    A node class that can be used in the ComfyUI project to load a hypernetwork model and apply it to a given model.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Returns the required input types for the HypernetworkLoader node.

        Returns:
        A dictionary containing the required input types.
        """
        return {"required": { "model": ("MODEL",),
                              "hypernetwork_name": (ldm_patched.utils.path_utils.get_filename_list("hypernetworks"), ),
                              "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_hypernetwork"

    CATEGORY = "loaders"

    def load_hypernetwork(self, model, hypernetwork_name, strength):
        """
        Loads the hypernetwork model from the given path and applies the patch to the model's attention layers.

        Args:
        model (Model): The model to which the hypernetwork model will be applied.
        hypernetwork_name (str): The name of the hypernetwork model to be loaded.
        strength (float): The strength value to be applied to the hypernetwork model.

        Returns:
        A tuple containing the updated model with the hypernetwork applied.
        """
        hypernetwork_path = ldm_patched.utils.path_utils.get_full_path("hypernetworks", hypernetwork_name)
        model_hypernetwork = model.clone()
        patch = load_hypernetwork_patch(hypernetwork_path, strength)
        if patch is not None:
            model_hypernetwork.set_model_attn1_patch(patch)
            model_hypernetwork.set_model_attn2_patch(patch)
        return (model_hypernetwork,)

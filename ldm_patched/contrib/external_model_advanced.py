# Import statements for external modules and classes
import ldm_patched.utils.path_utils 
import ldm_patched.modules.sd 
import ldm_patched.modules.model_sampling 
import torch

# Definition of the LCM class, a subclass of ldm_patched.modules.model_sampling.EPS
class LCM(ldm_patched.modules.model_sampling.EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        # Calculate the timestep using the provided sigma value
        timestep = self.timestep(sigma).view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        
        # Calculate the x0 value using the model_input and model_output values
        x0 = model_input - model_output * sigma
        
        # Define the sigma_data and scaled_timestep values
        sigma_data = 0.5
        scaled_timestep = timestep * 10.0 #timestep_scaling

        # Calculate the c_skip and c_out values using the sigma_data and scaled_timestep values
        c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5

        # Return the final value using the c_out and c_skip values
        return c_out * x0 + c_skip * model_input

# Definition of the ModelSamplingDiscrete class, a subclass of ldm_patched.modules.model_sampling.ModelSamplingDiscrete
class ModelSamplingDiscreteDistilled(ldm_patched.modules.model_sampling.ModelSamplingDiscrete):
    original_timesteps = 50

    def __init__(self, model_config=None):
        # Call the constructor of the parent class with the provided model_config argument
        super().__init__(model_config)

        # Calculate the skip_steps value using the num_timesteps and original_timesteps attributes
        self.skip_steps = self.num_timesteps // self.original_timesteps

        # Initialize the sigmas_valid tensor with the specified shape and data type
        sigmas_valid = torch.zeros((self.original_timesteps), dtype=torch.float32)
        for x in range(self.original_timesteps):
            # Set the value of sigmas_valid using the sigmas attribute and the current index
            sigmas_valid[self.original_timesteps - 1 - x] = self.sigmas[self.num_timesteps - 1 - x * self.skip_steps]

        # Call the set_sigmas method with the sigmas_valid tensor as the argument
        self.set_sigmas(sigmas_valid)

    def timestep(self, sigma):
        # Calculate the log_sigma tensor using the sigma argument
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        # Return the argmin value of the dists tensor along the specified dimension
        return (dists.abs().argmin(dim=0).view(sigma.shape) * self.skip_steps + (self.skip_steps - 1)).to(sigma.device)

    def sigma(self, timestep):
        # Calculate the t tensor using the timestep argument
        t = torch.clamp(((timestep.float().to(self.log_sigmas.device) - (self.skip_steps - 1)) / self.skip_steps).float(), min=0, max=(len(self.sigmas) - 1))
        # Calculate the low_idx and high_idx tensors using the t tensor
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        # Calculate the w tensor using the t tensor
        w = t.frac()
        # Calculate the log_sigma tensor using the low_idx, high_idx, and w tensors
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        # Return the exp value of the log_sigma tensor
        return log_sigma.exp().to(timestep.device)

# Definition of the rescale_zero_terminal_snr_sigmas function
def rescale_zero_terminal_snr_sigmas(sigmas):
    # Calculate the alphas_cumprod tensor using the sigmas argument
    alphas_cumprod = 1 / ((sigmas * sigmas) + 1)
    # Calculate the alphas_bar_sqrt tensor using the alphas_cumprod tensor
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store the alphas_bar_sqrt_0 and alphas_bar_sqrt_T values
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift the alphas_bar_sqrt tensor so the last timestep is zero
    alphas_bar_sqrt -= (alphas_bar_sqrt_T)

    # Scale the alphas_bar_sqrt tensor so the first timestep is back to the old value
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Calculate the alphas_bar tensor using the alphas_bar_sqrt tensor
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    # Set the last element of the alphas_bar tensor to a small value
    alphas_bar[-1] = 4.8973451890853435e-08
    # Return the ((1 - alphas_bar) / alphas_bar) ** 0.5 tensor
    return ((1 - alphas_bar) / alphas_bar) ** 0.5

# Definition of the ModelSamplingDiscrete class
class ModelSamplingDiscrete:
    # Class variable INPUT_TYPES, a dictionary with the required input types
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "sampling": (["eps", "v_prediction", "lcm"],),
                              "zsnr": ("BOOLEAN", {"default": False}),
                              }}

    # Class variable RETURN_TYPES, a list with the return types
    RETURN_TYPES = ("MODEL",)
    # Class variable FUNCTION, the name of the function
    FUNCTION = "patch"

    # Class variable CATEGORY, the category of the function
    CATEGORY = "advanced/model"

    # Definition of the patch function
    def patch(self, model, sampling, zsnr):
        # Create a copy of the model
        m = model.clone()

        # Determine the sampling_base and sampling_type classes based on the sampling argument
        sampling_base = ldm_patched.modules.model_sampling.ModelSamplingDiscrete
        if sampling == "eps":
            sampling_type = ldm_patched.modules.model_sampling.EPS
        elif sampling == "v_prediction":
            sampling_type = ldm_patched.modules.model_sampling.V_PREDICTION
        elif sampling == "lcm":
            sampling_base = ModelSamplingDiscreteDistilled
            sampling_type = LCM

        # Create a new class that is a subclass of both sampling_base and sampling_type
        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        # Create an instance of the ModelSamplingAdvanced class with the model.model.model_config argument
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        # If the zsnr argument is True, call the set_sigmas method with the result of the rescale_zero_terminal_snr_sigmas function
        if zsnr:
            model_sampling.set_sigmas(rescale_zero_terminal_snr_sigmas(model_sampling.sigmas))

        # Add a patch to the model object with the name "model_sampling" and the model_sampling instance
        m.add_object_patch("model_sampling", model_sampling)
        # Return the patched model
        return (m, )

# Definition of the ModelSamplingContinuousEDM class
class ModelSamplingContinuousEDM:
    # Class variable INPUT_TYPES, a dictionary with the required input types
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "sampling": (["v_prediction", "eps"],),
                              "sigma_max": ("FLOAT", {"default": 120.0, "min": 0.0, "max": 1000.0, "step":0.001, "round": False}),
                              "sigma_min": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1000.0, "step":0.001, "round": False}),
                              }}

    # Class variable RETURN_TYPES, a list with the return types
    RETURN_TYPES = ("MODEL",)
    # Class variable FUNCTION, the name of the function
    FUNCTION = "patch"

    # Class variable CATEGORY, the category of the function
    CATEGORY = "advanced/model"

    # Definition of the patch function
    def patch(self, model, sampling, sigma_max, sigma_min):
        # Create a copy of the model
        m = model.clone()

        # Determine the sampling_type class based on the sampling argument
        if sampling == "eps":
            sampling_type = ldm_patched.modules.model_sampling.EPS
        elif sampling == "v_prediction":
            sampling_type = ldm_patched.modules.model_sampling.V_PREDICTION

        # Create a new class that is a subclass of both ldm_patched.modules.model_sampling.ModelSamplingContinuousEDM and sampling_type
        class ModelSamplingAdvanced(ldm_patched.modules.model_sampling.ModelSamplingContinuousEDM, sampling_type):
            pass

        # Create an instance of the ModelSamplingAdvanced class with the model.model.model_config argument
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        # Call the set_sigma_range method with the sigma_min and sigma_max arguments
        model_sampling.set_sigma_range(sigma_min, sigma_max)
        # Add a patch to the model object with the name "model_sampling" and the model_sampling instance
        m.add_object_patch("model_sampling", model_sampling)
        # Return the patched model
        return (m, )

# Definition of the RescaleCFG class
class RescaleCFG:
    # Class variable INPUT_TYPES, a dictionary with the required input types
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "multiplier": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    # Class variable RETURN_TYPES, a list with the return types
    RETURN_TYPES = ("MODEL",)
    # Class variable FUNCTION, the name of the function
    FUNCTION = "patch"

    # Class variable CATEGORY,

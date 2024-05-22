import torch
import ldm_patched.modules.model_management  # Importing the necessary modules

def cast_bias_weight(s, input):   # Defining a function to cast bias and weight to the same device and dtype as input
    bias = None
    non_blocking = ldm_patched.modules.model_management.device_supports_non_blocking(input.device)
    if s.bias is not None:
        bias = s.bias.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)
    weight = s.weight.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)
    return weight, bias

class disable_weight_init:   # A class to disable weight initialization for certain layers
    class Linear(torch.nn.Linear):   # Defining a new Linear class that inherits from torch.nn.Linear
        ldm_patched_cast_weights = False   # A flag to control weight and bias casting
        def reset_parameters(self):   # Overriding the reset_parameters method to do nothing
            return None

        def forward_ldm_patched_cast_weights(self, input):   # Defining a new forward method for casting weights and bias
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):   # Defining the main forward method
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d):   # Defining a new Conv2d class that inherits from torch.nn.Conv2d
        ldm_patched_cast_weights = False   # A flag to control weight and bias casting
        def reset_parameters(self):   # Overriding the reset_parameters method to do nothing
            return None

        def forward_ldm_patched_cast_weights(self, input):   # Defining a new forward method for casting weights and bias
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):   # Defining the main forward method
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d):   # Defining a new Conv3d class that inherits from torch.nn.Conv3d
        ldm_patched_cast_weights = False   # A flag to control weight and bias casting
        def reset_parameters(self):   # Overriding the reset_parameters method to do nothing
            return None

        def forward_ldm_patched_cast_weights(self, input):   # Defining a new forward method for casting weights and bias
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):   # Defining the main forward method
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm):   # Defining a new GroupNorm class that inherits from torch.nn.GroupNorm
        ldm_patched_cast_weights = False   # A flag to control weight and bias casting
        def reset_parameters(self):   # Overriding the reset_parameters method to do nothing
            return None

        def forward_ldm_patched_cast_weights(self, input):   # Defining a new forward method for casting weights and bias
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

        def forward(self, *args, **kwargs):   # Defining the main forward method
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm):   # Defining a new LayerNorm class that inherits from torch.nn.LayerNorm
        ldm_patched_cast_weights = False   # A flag to control weight and bias casting
        def reset_parameters(self):   # Overriding the reset_parameters method to do nothing
            return None

        def forward_ldm_patched_cast_weights(self, input):   # Defining a new forward method for casting weights and bias
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        def forward(self, *args, **kwargs):   # Defining the main forward method
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    @classmethod   # A class method to create convolutional layers with arbitrary dimensions
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")

class manual_cast(disable_weight_init):   # A class that inherits from disable_weight_init and sets the casting flag to True
    class Linear(disable_weight_init.Linear):
        ldm_patched_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        ldm_patched_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        ldm_patched_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        ldm_patched_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        ldm_patched_cast_weights = True

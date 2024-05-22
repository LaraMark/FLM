import torch
from torch import nn

class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_updates=True):
        """
        Initialize the LitEma class, which is a PyTorch module that computes
        an exponential moving average (EMA) of the parameters in a given model.

        Args:
            model (nn.Module): The model to compute the EMA for.
            decay (float): The decay factor for the EMA. Default is 0.9999.
            use_num_updates (bool): Whether to use the number of updates as a
                factor in the decay. Default is True.
        """
        super().__init__()

        # Check that the decay is within the valid range
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        # Register a buffer to store the decay factor
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))

        # Register a buffer to store the number of updates
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if use_num_updates
        else torch.tensor(-1, dtype=torch.int))

        # Create a dictionary to map parameter names to shadow parameter names
        self.m_name2s_name = {}

        # Iterate over the named parameters in the model
        for name, p in model.named_parameters():
            if p.requires_grad:
                # Remove any '.' characters from the name
                s_name = name.replace('.', '')
                # Add an entry to the dictionary mapping the original name to the shadow name
                self.m_name2s_name.update({name: s_name})
                # Register a buffer for the shadow parameter
                self.register_buffer(s_name, p.clone().detach().data)

        # Initialize a list to store collected parameters
        self.collected_params = []

    def reset_num_updates(self):
        """
        Reset the number of updates to zero.
        """
        del self.num_updates
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))

    def forward(self, model):
        """
        Compute the EMA of the model parameters.

        Args:
            model (nn.Module): The model to compute the EMA for.

        Returns:
            None
        """
        decay = self.decay

        # If the number of updates is non-negative, increment it and adjust the decay
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        # Compute the EMA of the parameters
        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        """
        Copy the EMA of the parameters to the model.

        Args:
            model (nn.Module): The model to copy the EMA to.

        Returns:
            None
        """
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Store the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.

        Returns:
            None
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters.

        Returns:
            None
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

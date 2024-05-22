import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        x = self.linear(x)
        return x

# Create a custom Linear operation
class CustomLinear(nn.Linear):
    def forward(self, x):
        # Apply some custom transformation to the input
        x = super().forward(x)
        return x * 2

# Create a model using the custom Linear operation
model = MyModel()

# Patch the Linear operation with the custom implementation
with use_patched_ops(CustomLinear):
    # The model will now use the custom Linear operation
    y = model(x)

import math

class CA_layer(nn.Module):
    """Channel Attention layer that applies channel-wise attention to the input feature map.

    Args:
        channel (int): Number of channels in the input feature map.
        reduction (int, optional): Reduction ratio for the number of channels in the intermediate convolutional layer. Default: 16.

    Attributes:
        gap (nn.AdaptiveAvgPool2d): Global average pooling layer.
        fc (nn.Sequential): Sequential container for the intermediate convolutional layer.
    """
    def __init__(self, channel, reduction=16):
        super(CA_layer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global average pooling layer
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=(1, 1), bias=False),  # Intermediate convolutional layer with reduction in the number of channels
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=(1, 1), bias=False),  # Intermediate convolutional layer with restoration in the number of channels
            # nn.Sigmoid()  # Sigmoid activation function (commented out for flexibility)
        )

    def forward(self, x):
        """Applies the channel attention mechanism to the input feature map.

        Args:
            x (torch.Tensor): Input feature map with shape (batch_size, channel, height, width).

        Returns:
            torch.Tensor: Output feature map with the same shape as the input feature map, but with channel-wise attention applied.
        """
        y = self.fc(self.gap(x))  # Apply the intermediate convolutional layer to the global average pooled feature map
        return x * y.expand_as(x)  # Apply the channel-wise attention to the input feature map


class Simple_CA_layer(nn.Module):
    """Simple Channel Attention layer that applies channel-wise attention to the input feature map using a single convolutional layer.

    Args:
        channel (int): Number of channels in the input feature map.

    Attributes:
        gap (nn.AdaptiveAvgPool2d): Global average pooling layer.
        fc (nn.Conv2d): Convolutional layer for the intermediate layer.
    """
    def __init__(self, channel):
        super(Simple_CA_layer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global average pooling layer
        self.fc = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )  # Intermediate convolutional layer with the same number of channels

    def forward(self, x):
        """Applies the channel attention mechanism to the input feature map using a single convolutional layer.

        Args:
            x (torch.Tensor): Input feature map with shape (batch_size, channel, height, width).

        Returns:
            torch.Tensor: Output feature map with the same shape as the input feature map, but with channel-wise attention applied.
        """
        return x * self.fc(self.gap(x))  # Apply the intermediate convolutional layer to the global average pooled feature map and apply the channel-wise attention to the input feature map


class ECA_layer(nn.Module):
    """ECA module that applies channel-wise attention to the input feature map using a squeeze-and-excitation mechanism.

    Args:
        channel (int): Number of channels in the input feature map.

    Attributes:
        avg_pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        conv (nn.Conv1d): Convolutional layer for the intermediate layer.
    """
    def __init__(self, channel):
        super(ECA_layer, self).__init__()

        b = 1  # Hyperparameter for adaptive kernel size selection
        gamma = 2  # Hyperparameter for adaptive kernel size selection
        k_size = int(abs(math.log(channel, 2) + b) / gamma)  # Calculate the kernel size based on the number of channels
        k_size = k_size if k_size % 2 else k_size + 1  # Ensure the kernel size is odd

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling layer
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )  # Intermediate convolutional layer with adaptive kernel size
        # self.sigmoid = nn.Sigmoid()  # Sigmoid activation function (commented out for flexibility)

    def forward(self, x):
        """Applies the ECA mechanism to the input feature map.

        Args:
            x (torch.Tensor): Input feature map with shape (batch_size, channel, height, width).

        Returns:
            torch.Tensor: Output feature map with the same shape as the input feature map, but with channel-wise attention applied.
        """
        # x: input features with shape [b, c, h, w]
        # b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        # y = self.sigmoid(y)

        return x * y.expand_as(x)  # Apply the channel-wise attention to the input feature map


class ECA_MaxPool_layer(nn.Module):
    """ECA module that applies channel-wise attention to the input feature map using a squeeze-and-excitation mechanism with max pooling.

    Args:
        channel (int): Number of channels in the input feature map.

    Attributes:
        max_pool (nn.AdaptiveMaxPool2d): Global max pooling layer.
        conv (nn.Conv1d): Convolutional layer for the intermediate layer.
    """
    def __init__(self, channel):
        super(ECA_MaxPool_layer, self).__init__()

        b = 1  # Hyperparameter for adaptive kernel size selection
        gamma = 2  # Hyperparameter for adaptive kernel size selection
        k_size = int(abs(math.log(channel, 2) + b) / gamma)  # Calculate the kernel size based on the number of channels
        k_size = k_size if k_size % 2 else k_size + 1  # Ensure the kernel size is odd
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling layer
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )  # Intermediate convolutional layer with adaptive kernel size
        # self.sigmoid = nn.Sigmoid()  # Sigmoid activation function (commented out for flexibility)

    def forward(self, x):
        """Applies the ECA mechanism with max pooling to the input feature map.

        Args:
            x (torch.Tensor): Input feature map with shape (batch_size, channel, height, width).

        Returns:
            torch.Tensor: Output feature map with the same shape as the input feature map, but with channel-wise attention applied.
        """
        # x: input features with shape [b, c, h, w]
        # b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.max_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        # y = self.sigmoid(y)

        return x * y.expand_as(x)  # Apply the channel-wise attention to the input feature map

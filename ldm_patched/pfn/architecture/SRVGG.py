class SRVGGNetCompact(nn.Module):
    def __init__(
        self,
        state_dict,
        act_type: str = "prelu",
    ):
        # Initialize the SRVGGNetCompact class, inheriting from nn.Module
        # Store the activation type and state dictionary

        # Define the model architecture and sub-type
        self.model_arch = "SRVGG (RealESRGAN)"
        self.sub_type = "SR"

        self.act_type = act_type

        self.state = state_dict

        if "params" in self.state:
            self.state = self.state["params"]

        self.key_arr = list(self.state.keys())

        # Get the number of input channels, number of features, number of convolutions, and output channels
        self.in_nc = self.get_in_nc()
        self.num_feat = self.get_num_feats()
        self.num_conv = self.get_num_conv()
        self.out_nc = self.in_nc  # :(
        self.pixelshuffle_shape = None  # Defined in get_scale()
        self.scale = self.get_scale()

        # Set flags for supporting FP16 and BFP16
        self.supports_fp16 = True
        self.supports_bfp16 = True

        # Define the minimum size restriction
        self.min_size_restriction = None

        # Initialize the body of the network as a ModuleList
        self.body = nn.ModuleList()

        # Add the first convolution layer and activation layer
        self.body.append(nn.Conv2d(self.in_nc, self.num_feat, 3, 1, 1))
        self.body.append(self.get_activation(act_type))

        # Add the body structure with specified number of convolutions and activations
        for _ in range(self.num_conv):
            self.body.append(nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1))
            self.body.append(self.get_activation(act_type))

        # Add the last convolution layer
        self.body.append(nn.Conv2d(self.num_feat, self.pixelshuffle_shape, 3, 1, 1))

        # Add the PixelShuffle upsampler
        self.upsampler = nn.PixelShuffle(self.scale)

        # Load the state dictionary
        self.load_state_dict(self.state, strict=False)

    def get_num_conv(self) -> int:
        # Get the number of convolutions
        return (int(self.key_arr[-1].split(".")[1]) - 2) // 2

    def get_num_feats(self) -> int:
        # Get the number of features
        return self.state[self.key_arr[0]].shape[0]

    def get_in_nc(self) -> int:
        # Get the number of input channels
        return self.state[self.key_arr[0]].shape[1]

    def get_scale(self) -> int:
        # Calculate the upsampling factor (scale)
        self.pixelshuffle_shape = self.state[self.key_arr[-1]].shape[0]
        scale = math.sqrt(self.pixelshuffle_shape / self.out_nc)
        if scale - int(scale) > 0:
            print(
                "out_nc is probably different than in_nc, scale calculation might be wrong"
            )
        scale = int(scale)
        return scale

    def get_activation(self, act_type):
        # Get the activation layer based on the activation type
        if act_type == "relu":
            return nn.ReLU(inplace=True)
        elif act_type == "prelu":
            return nn.PReLU(num_parameters=self.num_feat)
        elif act_type == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # Define the forward pass
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)

        # Add the nearest upsampled image to the output for residual learning
        base = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        out += base

        return out

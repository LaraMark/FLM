class LatentFormat:
    # The `LatentFormat` class is the base class for all latent format types.
    # It contains common attributes and methods used by its subclasses.

    scale_factor = 1.0
    # `scale_factor` is a class attribute that determines the scaling factor
    # applied to the input latent during processing.

    latent_rgb_factors = None
    # `latent_rgb_factors` is a class attribute that holds the RGB factors
    # used for converting latent values to RGB colors. It is initialized as `None`
    # and is set in the subclasses.

    taesd_decoder_name = None
    # `taesd_decoder_name` is a class attribute that holds the name of the
    # TAESD decoder associated with the latent format. It is initialized as `None`
    # and is set in the subclasses.

    def process_in(self, latent):
        # The `process_in` method takes a latent value as input and scales it
        # using the `scale_factor` attribute before returning the result.
        return latent * self.scale_factor

    def process_out(self, latent):
        # The `process_out` method takes a latent value as input and inverse-scales it
        # using the `scale_factor` attribute before returning the result.
        return latent / self.scale_factor

class SD15(LatentFormat):
    # The `SD15` class is a subclass of `LatentFormat` that represents the SD15 latent format.

    def __init__(self, scale_factor=0.18215):
        # The `__init__` method initializes the `SD15` class with a default `scale_factor`
        # of 0.18215. It also sets the `latent_rgb_factors` and `taesd_decoder_name`
        # class attributes.

        self.scale_factor = scale_factor
        # `scale_factor` is an instance attribute that overrides the class attribute
        # with the same name. It determines the scaling factor applied to the input latent
        # during processing.

        self.latent_rgb_factors = [
            # `latent_rgb_factors` is a list of lists that holds the RGB factors
            # used for converting latent values to RGB colors.

            [ 0.3512,  0.2297,  0.3227],
            [ 0.3250,  0.4974,  0.2350],
            [-0.2829,  0.1762,  0.2721],
            [-0.2120, -0.2616, -0.7177]
        ]

        self.taesd_decoder_name = "taesd_decoder"
        # `taesd_decoder_name` is a string that holds the name of the TAESD decoder
        # associated with the SD15 latent format.

class SDXL(LatentFormat):
    # The `SDXL` class is a subclass of `LatentFormat` that represents the SDXL latent format.

    def __init__(self):
        # The `__init__` method initializes the `SDXL` class with default values
        # for the `scale_factor`, `latent_rgb_factors`, and `taesd_decoder_name`
        # class attributes.

        self.scale_factor = 0.13025
        self.latent_rgb_factors = [
            # `latent_rgb_factors` is a list of lists that holds the RGB factors
            # used for converting latent values to RGB colors.

            [ 0.3920,  0.4054,  0.4549],
            [-0.2634, -0.0196,  0.0653],
            [ 0.0568,  0.1687, -0.0755],
            [-0.3112, -0.2359, -0.2076]
        ]

        self.taesd_decoder_name = "taesdxl_decoder"
        # `taesd_decoder_name` is a string that holds the name of the TAESD decoder
        # associated with the SDXL latent format.

class SD_X4(LatentFormat):
    # The `SD_X4` class is a subclass of `LatentFormat` that represents the SD-X4 latent format.

    def __init__(self):
        # The `__init__` method initializes the `SD_X4` class with a default
        # `scale_factor` of 0.08333. The `latent_rgb_factors` and `taesd_decoder_name`
        # class attributes are not set, as they are not used in this latent format.

        self.scale_factor = 0.08333

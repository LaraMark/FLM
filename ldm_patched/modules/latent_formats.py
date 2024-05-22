class LatentFormat:
    # The `LatentFormat` class is the base class for all latent format types.
    # It contains common attributes and methods used by its derived classes.
    scale_factor: float = 1.0
    # `scale_factor` is a class attribute that represents the scaling factor applied to the latent.

    latent_rgb_factors: List[List[float]] = None
    # `latent_rgb_factors` is a class attribute that stores the RGB factors for converting latent to RGB.

    taesd_decoder_name: str = None
    # `taesd_decoder_name` is a class attribute that stores the name of the TAESD decoder.

    def process_in(self, latent):
        # The `process_in` method takes a latent as input and scales it using the `scale_factor`.
        return latent * self.scale_factor

    def process_out(self, latent):
        # The `process_out` method takes a latent as input and inverse scales it using the `scale_factor`.
        return latent / self.scale_factor

class SD15(LatentFormat):
    # The `SD15` class is a derived class of `LatentFormat` that represents the SD15 latent format.
    def __init__(self, scale_factor: float = 0.18215):
        # The `__init__` method initializes the `SD15` class with a default `scale_factor` of 0.18215.
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
            #   R        G        B
            [ 0.3512,  0.2297,  0.3227],
            [ 0.3250,  0.4974,  0.2350],
            [-0.2829,  0.1762,  0.2721],
            [-0.2120, -0.2616, -0.7177]
        ]
        self.taesd_decoder_name = "taesd_decoder"

class SDXL(LatentFormat):
    # The `SDXL` class is a derived class of `LatentFormat` that represents the SDXL latent format.
    def __init__(self):
        # The `__init__` method initializes the `SDXL` class with default values.
        self.scale_factor = 0.13025
        self.latent_rgb_factors = [
            #   R        G        B
            [ 0.3920,  0.4054,  0.4549],
            [-0.2634, -0.0196,  0.0653],
            [ 0.0568,  0.1687, -0.0755],
            [-0.3112, -0.2359, -0.2076]
        ]
        self.taesd_decoder_name = "taesdxl_decoder"

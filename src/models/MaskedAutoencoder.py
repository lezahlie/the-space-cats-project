from src.utils.common import pt, AttrDict
NN = pt.nn
F = NN.functional
from src.models.CNNEncoder import CNNEncoder
from src.models.CNNDecoder import CNNDecoder


class MaskedAutoencoder(NN.Module):
    def __init__(self, config: AttrDict):
        super(MaskedAutoencoder, self).__init__()

        # ==================================================
        # CONTRIBUTION START: Encoder Initialization Part 1
        # Contributor: Leslie Horace
        # ==================================================

        self.device = config.get('device', "cuda" if pt.cuda.is_available() else "cuda")
        self.random_seed = config.get('random_seed', 42)
 

        initial_seed = pt.initial_seed()
        if initial_seed != self.random_seed:
            raise RuntimeError(f"torch.initial_seed() returned {initial_seed}, but expected {self.random_seed}. Call SetupDevice.setup_torch_device(..., random_seed={self.random_seed}) first.")
        
        self.input_shape = config.get('input_shape', None)
        assert isinstance(self.input_shape, (tuple, list)) and len(self.input_shape) == 3, ("input_shape must be a tuple or list with format (C, H, W), where C=channels, H=height, and W=width")
        C, H, W = self.input_shape
        assert C > 0 and H > 3 and W == H, "input_shape must meet constraints: (C > 0, H > 3, W == H)"

        self.input_channels = C
        self.input_size = H

        batch_size = config.get('batch_size', None)
        if not isinstance(batch_size, int):
            raise ValueError(f"Missing entry for batch_size in config")
        elif batch_size < 4:
            raise ValueError(f"batch_size must be at least 4, not {batch_size}")
        
        self.hidden_factor = config.get('hidden_factor', 2.0)
        self.hidden_layers = config.get('hidden_layers', 3)
        self.hidden_dims = config.get('hidden_dims', 512)
        self.latent_dims = config.get('latent_dims', 256)
        self.ascending_channels = config.get('ascending_channels', True)

        assert self.hidden_factor > 1.0, f"hidden_factor must be > 1.0, not '{self.hidden_factor}'"
        assert self.hidden_layers > 0, f"hidden_layers must be at least 1, not '{self.hidden_layers}'"
        assert self.hidden_dims > 3, f"hidden_dims must be at least 4, not '{self.hidden_dims}'"
        assert self.latent_dims > 3, f"latent_dims must be at least 4, not '{self.latent_dims}'"
        
        self.activation_function = config.get("activation_function", "relu")
        # negative slope for leaky relu activation and weight initialization 
        self.negative_slope = config.get("negative_slope", 0.01)
        self.norm_layer = config.get("norm_layer", "none")
        self.conv_kernel = config.get("conv_kernel", 3)
        self.conv_stride = config.get("conv_stride", 1)

        assert self.norm_layer in {"none", "batch", "group"}, "norm_layer must be one of: 'none', 'batch', 'group'"
        assert self.activation_function in {"relu", "tanh", "leaky"}, "activation_function must be one of: 'relu', 'tanh', 'leaky'"
        assert self.conv_kernel % 2 == 1 and self.conv_kernel > 2, f"conv_kernel must be odd and at least 3 for symmetric padding, not '{self.conv_kernel}'"
        assert self.conv_stride > 0, f"self.conv_stride must be at least 1, not '{self.conv_stride}'"
        
        self.encoder = CNNEncoder(
            self.input_channels,
            self.input_size,
            hidden_layers = self.hidden_layers,
            hidden_dims= self.hidden_dims,
            hidden_factor= self.hidden_factor,
            latent_dims= self.latent_dims,
            activation_function = self.activation_function,
            negative_slope = self.negative_slope,
            norm_layer = self.norm_layer,
            conv_kernel=self.conv_kernel,
            conv_stride=self.conv_stride
        )

        self.encoder.to(device=self.device)

        # ==================================================
        # CONTRIBUTION END: Encoder Initialization Part 1
        # ==================================================

        # ==================================================
        # CONTRIBUTION START: Decoder Initialization Part 1
        # Contributor: Leslie Horace
        # ==================================================

        self.decoder = CNNDecoder(
            self.input_channels,
            self.input_size,
            hidden_layers = self.hidden_layers,
            hidden_dims= self.hidden_dims,
            hidden_factor= self.hidden_factor,
            latent_dims= self.latent_dims,
            activation_function = self.activation_function,
            negative_slope = self.negative_slope,
            norm_layer = self.norm_layer,
            conv_kernel=self.conv_kernel,
            conv_stride=self.conv_stride
        )

        self.decoder.to(device=self.device)
        # ==================================================
        # CONTRIBUTION END: Decoder Initialization Part 1
        # ==================================================

        self.to(self.device)

    def encode(self, x: pt.Tensor):
        """encodes (compresses) an input x to latent z

        Args: 
            x (tensor): input image
    
        Returns:
            tensor: latent z
        """
        # ==================================================
        # CONTRIBUTION START: Encoder Reconstruction 
        # Contributor: Wen Yu
        # ==================================================
        z = self.encoder(x)
        return z
        # ==================================================
        # CONTRIBUTION END: Encoder Reconstruction
        # ==================================================
    

    def decode(self, z: pt.Tensor):
        """decodes (reconstructs) a latent z to output y
        Args: 
            z (tensor): output image 
    
        Returns:
            tensor: reconstructed y
        """
        # ==================================================
        # CONTRIBUTION START: Decoder Reconstruction
        # Contributor: Leslie Horace
        # ==================================================
        y = self.decoder(z)
        return y
        # ==================================================
        # CONTRIBUTION END: Decoder Reconstruction
        # ==================================================



def test_main(args):
    from src.utils.device import SetupDevice
    from src.utils.logger import set_logger_level, get_logger
    from src.utils.config import DEFAULT_TRAIN_CONFIG
    set_logger_level(10)
    logger = get_logger()

    device = SetupDevice.setup_torch_device(
        args.num_cores,
        args.cpu_device_only,
        args.gpu_device_list,
        args.gpu_memory_fraction,
        args.random_seed
    )

    batch_size = 32
    input_channels = 5
    input_size = 64
    latent_dims = 128

    batch_input_shape = (batch_size, input_channels, input_size, input_size)
    expected_encoder_shape = (batch_size, latent_dims)

    config = AttrDict(DEFAULT_TRAIN_CONFIG)
    config.input_shape = batch_input_shape[1:]
    model = MaskedAutoencoder(config)

    dummy_input_batch = pt.randn(batch_input_shape, device=device)

    logger.debug(f"Batch Input shape: {dummy_input_batch.shape}")
    logger.debug(f"[CNNEncoder]:\nEncoder Input Layer: {model.encoder.input_layer}"
                f"\nEncoder Hidden Layers: {model.encoder.encoder_layers}"
                f"\nEncoder Output Layer: {model.encoder.output_layer}")

    encoder_output = model.encode(dummy_input_batch)
    if encoder_output == NotImplemented:
        logger.debug("CNNEncoder not implemented yet")
        return
    logger.debug(f"Encoder output shape: {encoder_output.shape}, expected_shape: {expected_encoder_shape}")

    logger.debug(f"Latent input shape: {encoder_output.shape}")
    logger.debug(f"[CNNDecoder]:\nDecoder Input Layer: {model.decoder.input_layer}"
                f"\nDecoder Hidden Layers: {model.decoder.decoder_layers}"
                f"\nDecoder Output Layer: {model.decoder.output_layer}")

    decoder_output = model.decode(encoder_output)
    if decoder_output == NotImplemented:
        logger.debug("CNNDecoder not implemented yet")
        return
    
    logger.debug(f"Decoder output shape: {decoder_output.shape}, expected_shape: {batch_input_shape}")


if __name__ == "__main__":
    from src.utils.logger import init_shared_logger
    from src.utils.common import pt, AttrDict
    init_shared_logger(__file__, log_stdout=True, log_stderr=True)
    fake_args = AttrDict(
        num_cores = 1,
        cpu_device_only = False,
        gpu_device_list = [0],
        gpu_memory_fraction = 0.5,
        random_seed = 42
    )

    test_main(fake_args)

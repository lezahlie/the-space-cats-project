from src.utils.common import pt
NN = pt.nn
F = NN.functional


class CNNEncoder(NN.Module):
    def __init__(
        self, 
        input_channels: int,
        input_size: int,
        conv_kernel: int=3,
        conv_stride: int=2,
        hidden_layers: int=3, 
        hidden_dims: int=128, 
        hidden_factor: float=2.0,
        latent_dims: int=64, 
        activation_function: str = "relu",
        negative_slope: float = 0.01,
        norm_layer: str = "none",
        ascending_channels: bool = True
    ): 
        super(CNNEncoder, self).__init__()

        # ==================================================
        # CONTRIBUTION START: Encoder Initialization Part 2
        # Contributor: Leslie Horace
        # ==================================================

        # initialize input channels and dims
        self.input_channels = input_channels
        # assume inputs are square so input height == width
        self.input_size = input_size

        # initialize layer settings
        self.hidden_layers = hidden_layers
        self.hidden_dims = hidden_dims
        self.latent_dims = latent_dims
        self.hidden_factor = hidden_factor
        self.ascending_channels = ascending_channels

        # initialize other network settings
        self.norm_layer = norm_layer
        self.activation_function = activation_function
        self.negative_slope = negative_slope

        # initialize kernel settings
        if conv_kernel % 2 == 0 or conv_kernel < 3:
            raise ValueError(f"conv_kernel must be odd and at least 3 for symmetric padding, not '{conv_kernel}'")
        
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        # symmetric padding that works with stride >= 1 and odd kernels only
        self.conv_padding = self.conv_kernel  // 2
        # compute hidden layer dims and validate
        self.hidden_per_layer = [int(self.hidden_dims / (self.hidden_factor ** i)) for i in range(self.hidden_layers)]
        if self.hidden_per_layer[-1] < 4:
            raise ValueError(f"hidden_factor={self.hidden_factor} is too large to reduce hidden_dims={self.hidden_dims} across hidden_layers={self.hidden_layers}\nComputed hidden dims: {self.hidden_per_layer}")
        
        # if ascending_channels is True: encoder channels INCREASE with depth (reverse hidden dims per layer)
        # Otherwise: encoder channels DECEASE with depth (leave as is)
        if self.ascending_channels:
            self.hidden_per_layer = self.hidden_per_layer[::-1]

        compute_output_size = lambda x: ((x - self.conv_kernel + 2 * self.conv_padding) // self.conv_stride) + 1


        # initialize input layer: Conv2D => GroupNorm/BatchNorm2D/None => Activation
        self.input_layer = NN.Sequential(
            NN.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.hidden_per_layer[0],
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding,
            ),
            self._get_norm_layer(self.hidden_per_layer[0]),
            self._get_activation(),
        )
        # compute output size based on input size
        output_size = compute_output_size(self.input_size)

        # initialize encoder layers: input_layer => encoder_layers
        self.encoder_layers = NN.ModuleList()
        for i in range(1, self.hidden_layers):
            in_chan = self.hidden_per_layer[i - 1]
            out_chan = self.hidden_per_layer[i]

            # hidden layer: Conv2D => GroupNorm/BatchNorm2D/None => Activation
            self.encoder_layers.append(
                NN.Sequential(
                    NN.Conv2d(
                        in_channels=in_chan,
                        out_channels=out_chan,
                        kernel_size=self.conv_kernel,
                        stride=self.conv_stride,
                        padding=self.conv_padding,
                    ),
                    self._get_norm_layer(out_chan),
                    self._get_activation(),
                )
            )
            output_size = compute_output_size(output_size)


        # init encoder output layer: project hidden_output => latent_z
        # 1x1 conv2d to reduce spatial size while retaining spatial detail
        self.output_layer = NN.Sequential(
            NN.Conv2d(
                in_channels=self.hidden_per_layer[-1],
                out_channels=self.latent_dims,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        # weight initialization — apply AFTER all layers are built
        self.apply(self._init_weights)
        # ==================================================
        # CONTRIBUTION END: Encoder Initialization Part 2
        # ==================================================

    def _get_activation(self):
        # ==================================================
        # CONTRIBUTION START: Encoder Activation Initialization
        # Contributor: Leslie Horace
        # ==================================================
        # initialize element wise activation
        if self.activation_function == "relu":
            return NN.ReLU()
        elif self.activation_function == "leaky":
            return NN.LeakyReLU(negative_slope=self.negative_slope)
        elif self.activation_function == "tanh":
            return NN.Tanh()
        else:
            raise ValueError(f"unsupported activation_function '{self.activation_function}'")
        
        # ==================================================
        # CONTRIBUTION END: Encoder Activation Initialization
        # ==================================================

    def _get_norm_layer(self, out_chan):
        # ==================================================
        # CONTRIBUTION START: Encoder Normalization Initialization
        # Contributor: Leslie Horace
        # ==================================================
        if self.norm_layer == "batch":
            return NN.BatchNorm2d(out_chan)
        elif self.norm_layer == "group":
            num_groups = min(8, out_chan)
            while out_chan % num_groups != 0:
                num_groups -= 1
            return NN.GroupNorm(num_groups, out_chan)
        return NN.Identity()
    
        # ==================================================
        # CONTRIBUTION END: Norm Layer Initialization Helper
        # ==================================================

    def _init_weights(self, module):
        # ==================================================
        # CONTRIBUTION START: Encoder Weight Initialization
        # Contributor: Leslie Horace
        # ==================================================
        if not isinstance(module, (NN.Conv2d, NN.ConvTranspose2d, NN.Linear)):
            return
        if self.activation_function == "relu":
            NN.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif self.activation_function == "leaky":
            NN.init.kaiming_normal_(module.weight, a=self.negative_slope, nonlinearity="leaky_relu")
        elif self.activation_function == "tanh":
            NN.init.xavier_normal_(module.weight)
        if module.bias is not None:
            NN.init.zeros_(module.bias)
            
        # ==================================================
        # CONTRIBUTION END: Encoder Weight Initialization
        # ==================================================
    
    def forward(self, x):
        # ==================================================
        # CONTRIBUTION START: Encoder Forward Pass
        # Contributor: Wen Yu
        # ==================================================
        x = self.input_layer(x)
        for layer in self.encoder_layers:
            x = layer(x)
        z = self.output_layer(x)
        return z
        # ==================================================
        # CONTRIBUTION END: Encoder Forward Pass
        # ==================================================


def test_main(args):
    from src.utils.device import SetupDevice
    from src.utils.logger import set_logger_level, get_logger

    set_logger_level(10)
    logger = get_logger()

    device = SetupDevice.setup_torch_device(
        args.num_cores,
        args.cpu_device_only,
        args.gpu_device_list,
        args.gpu_memory_fraction,
        args.random_seed
    )

    input_channels = 5
    input_size = 64
    batch_size = 20
    latent_dims = 64
    hidden_layers= 3
    conv_stride = 2

    spatial_size = input_size // (conv_stride ** hidden_layers)
    batch_input_shape = (batch_size, input_channels, input_size, input_size)
    expected_encoder_shape = (batch_size, latent_dims, spatial_size, spatial_size)

    model = CNNEncoder(
        input_channels, 
        input_size,
        hidden_layers = hidden_layers,
        latent_dims = latent_dims
    )


    dummy_input_batch = pt.randn(batch_input_shape)
    logger.debug(f"Batch Input shape: {dummy_input_batch.shape}")

    logger.debug(f"[CNNEncoder]:\nEncoder Input Layer: {model.input_layer}"
                    f"\nEncoder Hidden Layers: {model.encoder_layers}"
                    f"\nEncoder Output Layer: {model.output_layer}")

    encoder_output = model(dummy_input_batch)
    if encoder_output == NotImplemented:
        logger.debug("CNNEncoder not implemented yet")
        return

    logger.debug(f"Encoder output shape: {encoder_output.shape}, expected_shape: {expected_encoder_shape}")


if __name__ == "__main__":
    from src.utils.logger import init_shared_logger
    from src.utils.common import pt, SimpleNamespace
    init_shared_logger(__file__, log_stdout=True, log_stderr=True)
    fake_args = SimpleNamespace(
        num_cores = 1,
        cpu_device_only = False,
        gpu_device_list = [0],
        gpu_memory_fraction = 0.5,
        random_seed = 42
    )

    test_main(fake_args)

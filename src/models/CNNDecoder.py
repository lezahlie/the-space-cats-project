from src.utils.common import pt
NN = pt.nn
F = NN.functional

class CNNDecoder(NN.Module):
    def __init__(
        self, 
        input_channels: int,
        input_size: int,
        conv_kernel: int=3,
        conv_stride: int=1,
        hidden_layers: int=3, 
        hidden_dims: int=256, 
        hidden_factor: float=2.0,
        latent_dims: int=128, 
        activation_function: str = "relu",
        negative_slope: float = 0.01,
        norm_layer: str = "none",
        ascending_channels: bool = False
    ): 
        super(CNNDecoder, self).__init__()
        
        # ==================================================
        # CONTRIBUTION START: Decoder Initialization Part 2
        # Contributor: Wen Yu
        # ==================================================
        # store settings (mirror encoder)
        self.input_channels = input_channels
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_dims = hidden_dims
        self.hidden_factor = hidden_factor
        self.latent_dims = latent_dims 
        self.activation_function = activation_function
        self.negative_slope = negative_slope
        self.norm_layer = norm_layer
        self.ascending_channels = ascending_channels

        # initialize kernel settings
        if conv_kernel % 2 == 0 or conv_kernel < 3:
            raise ValueError(f"conv_kernel must be odd and at least 3 for symmetric padding, not '{conv_kernel}'")
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        # symmetric padding that works with stride >= 1 and odd kernels only
        self.conv_padding = self.conv_kernel  // 2

        if self.conv_kernel % 2 == 0 or self.conv_kernel < 3:
            raise ValueError(f"conv_kernel must be odd and at least 3 for symmetric padding, not '{self.conv_kernel}'")
        #   mirror encoder: hidden dims depend on ascending_channels
        # encoder: [32, 64, 128], decoder: [128, 64, 32]
        self.hidden_per_layer = [
            int(self.hidden_dims / (self.hidden_factor ** i)) 
            for i in range(self.hidden_layers)
        ]
        
        # if ascending_channels is True: decoder channels DECEASE with depth (leave as is)
        # Otherwise: decoder channels INCREASE with depth (reverse hidden dims per layer)
        if not self.ascending_channels:
            self.hidden_per_layer = self.hidden_per_layer[::-1]

        # compute the spatial size at the bottleneck
        # (same formula as encoder, applied hidden_layers times)
        compute_output_size = lambda x: ((x - self.conv_kernel + 2 * self.conv_padding) // self.conv_stride) + 1

        bottleneck_size = self.input_size
        self.encoder_sizes = [self.input_size]
        for _ in range(self.hidden_layers):
            bottleneck_size = compute_output_size(bottleneck_size)
            self.encoder_sizes.append(bottleneck_size)
    
        self.reduced_channels = int(max(self.hidden_per_layer[0] // self.hidden_factor, 4))
        self.bottleneck_size = bottleneck_size
        self.bottleneck_HxW = bottleneck_size ** 2
        # input layer: Linear → Unflatten
        # latent z (batch, latent_dims) → (batch, C, H, W) at bottleneck
        self.input_layer = NN.Sequential(
            NN.Linear(
                in_features=self.latent_dims,
                out_features=self.reduced_channels * self.bottleneck_HxW,
            ),
            NN.Unflatten(
                dim=1, 
                unflattened_size=(self.reduced_channels, self.bottleneck_size, self.bottleneck_size)
            ),
            NN.Conv2d(
                in_channels=self.reduced_channels,
                out_channels=self.hidden_per_layer[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        # hidden layers: ConvTranspose2d × (hidden_layers - 1)
        current_size = self.bottleneck_size
        self.decoder_layers = NN.ModuleList()
        for i in range(1, self.hidden_layers):
            in_chan = self.hidden_per_layer[i - 1]
            out_chan = self.hidden_per_layer[i]

            target_size = self.encoder_sizes[-(i+1)]
            output_padding = self._get_output_padding(current_size, target_size)
            self.decoder_layers.append(
                NN.Sequential(
                    NN.ConvTranspose2d(
                        in_channels=in_chan,
                        out_channels=out_chan,
                        kernel_size=self.conv_kernel,
                        stride=self.conv_stride,
                        padding=self.conv_padding,
                        output_padding=output_padding
                    ),
                    self._get_norm_layer(out_chan),
                    self._get_activation(),
                )
            )
            current_size = target_size

        final_output_padding = self._get_output_padding(current_size, self.input_size)
        # output layer: project back to original image channels
        # no activation after — let loss function decide (e.g. sigmoid outside)
        self.output_layer = NN.Sequential(
            NN.ConvTranspose2d(
                in_channels=self.hidden_per_layer[-1],
                out_channels=self.input_channels,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding,
                output_padding=final_output_padding,
            ),
        )

        # weight initialization — apply AFTER all layers are built
        self.apply(self._init_weights)
        # ==================================================
        # CONTRIBUTION END: Decoder Initialization Part 2
        # ==================================================


    def _get_output_padding(self, in_size: int, target_size: int) -> int:
        output_padding = target_size - (
            (in_size - 1) * self.conv_stride
            - 2 * self.conv_padding
            + self.conv_kernel
        )
        if not (0 <= output_padding < self.conv_stride):
            raise ValueError(
                f"Invalid output_padding={output_padding} for "
                f"in_size={in_size}, target_size={target_size}, "
                f"kernel={self.conv_kernel}, stride={self.conv_stride}, "
                f"padding={self.conv_padding}"
            )
        return output_padding

    def _get_activation(self):
        if self.activation_function == "relu":
            return NN.ReLU()
        elif self.activation_function == "leaky":
            return NN.LeakyReLU(negative_slope=self.negative_slope)
        elif self.activation_function == "tanh":
            return NN.Tanh()
        else:
            raise ValueError(f"unsupported activation_function '{self.activation_function}'")

    def _get_norm_layer(self, out_chan):
        if self.norm_layer == "batch":
            return NN.BatchNorm2d(out_chan)
        elif self.norm_layer == "group":
            num_groups = min(8, out_chan)
            while out_chan % num_groups != 0:
                num_groups -= 1
            return NN.GroupNorm(num_groups, out_chan)
        return NN.Identity()
    

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

    def forward(self, y):
        # ==================================================
        # CONTRIBUTION START: Decoder Forward Pass
        # Contributor: Wen Yu
        # ==================================================
        x = self.input_layer(y)
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
        # ==================================================
        # CONTRIBUTION END: Decoder Forward Pass
        # ==================================================
        # return NotImplemented


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

    batch_size = 20
    input_channels = 5
    input_size = 64
    latent_dims = 128

    # simulate encoder's output (latent vector)
    dummy_latent = pt.randn(batch_size, latent_dims)
    expected_output_shape = (batch_size, input_channels, input_size, input_size)

    model = CNNDecoder(
        input_channels, 
        input_size, 
        conv_kernel=3,
        conv_stride=1,
        hidden_layers = 3,
        hidden_dims=128,
        hidden_factor=2.0,
        latent_dims=latent_dims,
        activation_function="leaky",
        negative_slope= 0.01,
        norm_layer="group",
        ascending_channels=True
    )

    logger.debug(f"Latent input shape: {dummy_latent.shape}")
    logger.debug(f"[CNNDecoder]:\nDecoder Input Layer: {model.decoder.input_layer}"
                f"\nDecoder Hidden Layers: {model.decoder.decoder_layers}"
                f"\nDecoder Output Layer: {model.decoder.output_layer}")

    decoder_output = model(dummy_latent)
    logger.debug(f"Decoder output shape: {decoder_output.shape}, expected_shape: {expected_output_shape}")

    assert decoder_output.shape == pt.Size(list(expected_output_shape)), \
        f"Shape mismatch! Got {decoder_output.shape}, expected {expected_output_shape}"
    logger.debug("Decoder output shape matches expected shape!")


if __name__ == "__main__":
    from src.utils.logger import init_shared_logger
    from src.utils.common import pt, SimpleNamespace
    init_shared_logger(__file__, log_stdout=True, log_stderr=True)
    fake_args = SimpleNamespace(
        num_cores=1,
        cpu_device_only=False,
        gpu_device_list=[0],
        gpu_memory_fraction=0.5,
        random_seed=42
    )
    test_main(fake_args)
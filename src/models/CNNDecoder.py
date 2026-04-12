from src.utils.common import pt
NN = pt.nn
F = NN.functional

class CNNDecoder(NN.Module):
    def __init__(
        self, 
        input_dims: int,    # latent_dims
        input_channels: int,
        input_size: int,
        conv_kernel: int,
        conv_stride: int,
        apply_groupnorm: bool,
        device: pt.DeviceObjType,
        generator: pt.Generator,
        hidden_layers: int=3, 
        hidden_dims: int=128, 
        hidden_factor: float=2.0,
        latent_dims: int=64, 
        activation_name: str = "relu",
        negative_slope: float = 0.01,
        apply_batchnorm: bool = False,
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
        self.latent_dims = input_dims        # input_dims == encoder latent_dims
        self.activation_name = activation_name
        self.negative_slope = negative_slope
        self.apply_batchnorm = apply_batchnorm
        self.apply_groupnorm = apply_groupnorm
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_padding = (self.conv_kernel - self.conv_stride) // 2

        # mirror encoder: hidden dims go small → large
        # encoder: [128, 64, 32], decoder: [32, 64, 128]
        self.hidden_per_layer = [
            int(self.hidden_dims / (self.hidden_factor ** i)) 
            for i in range(self.hidden_layers)
        ][::-1]  # reverse to mirror encoder

        # compute the spatial size at the bottleneck
        # (same formula as encoder, applied hidden_layers times)
        compute_output_size = lambda x: ((x - self.conv_kernel + 2 * self.conv_padding) // self.conv_stride) + 1
        bottleneck_size = self.input_size
        for _ in range(self.hidden_layers):
            bottleneck_size = compute_output_size(bottleneck_size)
        self.bottleneck_size = bottleneck_size
        self.bottleneck_HxW = bottleneck_size ** 2

        # input layer: Linear → Unflatten
        # latent z (batch, latent_dims) → (batch, C, H, W) at bottleneck
        self.input_layer = NN.Sequential(
            NN.Linear(
                in_features=self.latent_dims,
                out_features=self.hidden_per_layer[0] * self.bottleneck_HxW,
            ),
            NN.Unflatten(
                dim=1, 
                unflattened_size=(self.hidden_per_layer[0], self.bottleneck_size, self.bottleneck_size)
            ),
            self._get_activation(),
        )

        # hidden layers: ConvTranspose2d × (hidden_layers - 1)
        self.decoder_layers = NN.ModuleList()
        for i in range(1, self.hidden_layers):
            in_chan = self.hidden_per_layer[i - 1]
            out_chan = self.hidden_per_layer[i]
            self.decoder_layers.append(
                NN.Sequential(
                    NN.ConvTranspose2d(
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

        # output layer: project back to original image channels
        # no activation after — let loss function decide (e.g. sigmoid outside)
        self.output_layer = NN.Sequential(
            NN.ConvTranspose2d(
                in_channels=self.hidden_per_layer[-1],
                out_channels=self.input_channels,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding,
            ),
        )
        # ==================================================
        # CONTRIBUTION END: Decoder Initialization Part 2
        # ==================================================

    def _get_activation(self):
        if self.activation_name == "relu":
            return NN.ReLU()
        elif self.activation_name == "leaky":
            return NN.LeakyReLU(negative_slope=self.negative_slope)
        elif self.activation_name == "tanh":
            return NN.Tanh()
        else:
            raise ValueError(f"unsupported activation_name '{self.activation_name}'")

    def _get_norm_layer(self, out_chan):
        if self.apply_batchnorm:
            return NN.BatchNorm2d(out_chan)
        elif self.apply_groupnorm:
            num_groups = min(8, out_chan)
            while out_chan % num_groups != 0:
                num_groups -= 1
            return NN.GroupNorm(num_groups, out_chan)
        return NN.Identity()

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
        input_dims=latent_dims,
        input_channels=input_channels,
        input_size=input_size,
        hidden_layers=3,
        hidden_dims=128,
        hidden_factor=2.0,
        latent_dims=latent_dims,
        activation_name="relu",
        negative_slope=0.01,
        apply_batchnorm=False,
        apply_groupnorm=False,
        conv_kernel=5,
        conv_stride=1,
        device=device,
        generator=None,
    )

    logger.debug(f"Latent input shape: {dummy_latent.shape}")
    logger.debug(f"Decoder Input Layer:\n{model.input_layer}")
    logger.debug(f"Decoder Hidden Layers:\n{model.decoder_layers}")
    logger.debug(f"Decoder Output Layer:\n{model.output_layer}")

    decoder_output = model(dummy_latent)
    logger.debug(f"Decoder output shape: {decoder_output.shape}, expected_shape: {expected_output_shape}")

    assert decoder_output.shape == pt.Size(list(expected_output_shape)), \
        f"Shape mismatch! Got {decoder_output.shape}, expected {expected_output_shape}"
    logger.debug("✅ Decoder output shape matches expected shape!")


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
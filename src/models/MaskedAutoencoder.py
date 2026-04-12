from src.utils.common import pt
NN = pt.nn
F = NN.functional
from models.CNNEncoder import CNNEncoder
from models.CNNDecoder import CNNDecoder

class MaskedAutoencoder(NN.Module):
    def __init__(self, model_params: dict):
        super(MaskedAutoencoder, self).__init__()

        # ==================================================
        # CONTRIBUTION START: Encoder Initialization Part 1
        # Contributor: Leslie Horace
        # ==================================================

        self.device = model_params.get('device', "cuda" if pt.cuda.is_available() else "cuda")
        self.random_seed = model_params.get('random_seed', 42)

        initial_seed = pt.initial_seed()
        if initial_seed != self.random_seed:
            raise RuntimeError(f"torch.initial_seed() returned {initial_seed}, but expected {self.random_seed}. Call SetupDevice.setup_torch_device(..., random_seed={self.random_seed}) first.")
        
        self.input_shape = model_params.get('input_shape', None)
        assert isinstance(self.input_shape, (tuple, list)) and len(self.input_shape) == 3, ("input_shape must be a tuple or list with format (C, H, W), where C=channels, H=height, and W=width")
        C, H, W = self.input_shape
        assert C > 0 and H > 3 and W == H, "input_shape must meet constraints: (C > 0, H > 3, W == H)"

        self.input_channels = C
        self.input_size = H

        self.hidden_factor = model_params.get('hidden_factor', 2.0)
        self.hidden_layers = model_params.get('hidden_layers', 3)
        self.hidden_dims = model_params.get('hidden_dims', 256)
        self.latent_dims = model_params.get('latent_dims', 128)

        assert self.hidden_factor > 1.0, f"hidden_factor must be > 1.0, not '{self.hidden_factor}'"
        assert self.hidden_layers > 0, f"hidden_layers must be at least 1, not '{self.hidden_layers}'"
        assert self.hidden_dims > 3, f"hidden_dims must be at least 4, not '{self.hidden_dims}'"
        assert self.latent_dims > 3, f"latent_dims must be at least 4, not '{self.latent_dims}'"

        self.activation_name = model_params.get("activation_function", "relu")
        # negative slope for leaky relu activation and weight initialization 
        self.negative_slope = model_params.get("negative_slope", 0.01)
        self.apply_batchnorm = model_params.get("apply_batchnorm", 0.01)
        self.apply_groupnorm = model_params.get("apply_groupnorm", 0.01)
        self.conv_kernel = model_params.get("conv_kernel", 3)
        self.conv_stride = model_params.get("conv_stride", 1)

        assert self.activation_name in {"relu", "tanh", "leaky"}, "activation_function must be one of: 'relu', 'tanh', 'leaky'"
        assert not (self.apply_batchnorm and self.apply_groupnorm), "apply_batchnorm and apply_groupnorm cannot both be True"
        assert self.conv_kernel % 2 == 1 and self.conv_kernel > 2, f"self.conv_kernel must be odd and at least 3, not '{self.conv_kernel}'"
        assert self.conv_stride > 0, f"self.conv_stride must be at least 1, not '{self.conv_stride}'"

        # @todo need to implement this later
        self.mask_ratio = model_params.get('mask_ratio', 0.0)
        assert 0.0 <= self.mask_ratio <= 1.0, f"mask_ratio must be in range [0.0, 1.0], not '{self.mask_ratio}'"

        self.encoder = CNNEncoder(
            self.input_channels,
            self.input_size,
            hidden_layers = self.hidden_layers,
            hidden_dims= self.hidden_dims,
            hidden_factor= self.hidden_factor,
            latent_dims= self.latent_dims,
            activation_name = self.activation_name,
            negative_slope = self.negative_slope,
            apply_batchnorm = self.apply_batchnorm,
            apply_groupnorm = self.apply_groupnorm,
            conv_kernel=self.conv_kernel,
            conv_stride=self.conv_stride
        )

        self.encoder.to(device=self.device)

        # ==================================================
        # CONTRIBUTION END: Encoder Initialization Part 1
        # ==================================================


        # ==================================================
        # CONTRIBUTION START: Decoder Initialization Part 1
        # Contributor: <First> <Last>
        # ==================================================

        # ==================================================
        # CONTRIBUTION END: Decoder Initialization Part 1
        # ==================================================

        self.to(self.device)

    def encode(self, x):
        """encodes (compresses) an input x to latent z

        Args: 
            x (tensor): input image
    
        Returns:
            tensor: latent z
        """
        z = None
        # ==================================================
        # CONTRIBUTION START: Encoder Reconstruction 
        # Contributor: Wen Yu
        # ==================================================
        z = self.encoder(x)
        return z
        # ==================================================
        # CONTRIBUTION END: Encoder Reconstruction
        # ==================================================
    


    def decode(self, y):
        """decodes (reconstructs) a latent z to output y
        Args: 
            y (tensor): output image 
    
        Returns:
            tensor: reconstructed x
        """
        x = None
        # ==================================================
        # CONTRIBUTION START: Decoder Reconstruction
        # Contributor: <First> <Last>
        # ==================================================

    
        # ==================================================
        # CONTRIBUTION END: Decoder Reconstruction
        # ==================================================
        return NotImplemented
from src.utils.common import pt
NN = pt.nn
F = NN.functional

class CNNDecoder(NN.Module):
    def __init__(
        self, 
        input_dims: int,
        device: pt.DeviceObjType,
        generator: pt.Generator,
        hidden_layers: int=3, 
        hidden_dims: int=128, 
        hidden_factor: float=2.0,
        latent_dims: int=64, 
        activation_name: str = "relu",
        negative_slope: float = 0.01,
        apply_batchnorm: bool = False
    ): 
        super(CNNDecoder, self).__init__()
        
        # ==================================================
        # CONTRIBUTION START: Decoder Initialization Part 2
        # Contributor: <First> <Last>
        # ==================================================

        # ==================================================
        # CONTRIBUTION END: Decoder Initialization Part 2
        # ==================================================

    def forward(self, y):

        # ==================================================
        # CONTRIBUTION START: Decoder Forward Pass
        # Contributor: <First> <Last>
        # ==================================================

        # ==================================================
        # CONTRIBUTION END: Decoder Forward Pass
        # ==================================================
        return NotImplemented

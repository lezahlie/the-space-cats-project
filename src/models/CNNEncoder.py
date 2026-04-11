import torch as pt
NN = pt.nn
F = NN.functional

class CNNEncoder(NN.Module):
    def __init__(
        self, 
        hidden_layers: int=3, 
        hidden_dims: int=128, 
        latent_dims: int=64, 
        hidden_factor: float=2.0,
        activation: pt.Module=NN.ReLU, 
    ): super(CNNEncoder, self).__init__()

    """initialize the encoder
    """

    def forward(self):
        return NotImplementedError()

    

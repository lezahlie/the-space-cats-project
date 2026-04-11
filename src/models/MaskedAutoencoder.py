import torch as pt
NN = pt.nn
F = NN.functional
from models.CNNEncoder import CNNEncoder
from models.CNNDecoder import CNNDecoder

class MaskedAutoEncoder(NN.Module):
    def __init__(self, model_params: dict):
        super(MaskedAutoEncoder, self).__init__()

        """initialize the full autoencoder
        """

        self.hidden_factor = model_params.get('hidden_factor', 2.0)
        self.hidden_layers = model_params.get('hidden_layers', 3)
        self.hidden_dims = model_params.get('hidden_dims', 128)
        self.latent_dims = model_params.get('latent_dims', 68)
        self.activation = model_params.get('activation_function', "relu")
        
        valid_activations = {"relu", "leaky", "tanh"}
        if self.activation not in valid_activations:
            raise ValueError(f"activation '{self.activation} must be in {valid_activations}")
    
        self.hidden_dims_per_layer = [self.hidden_dims // self.hidden_factor for _ in self.hidden_layers]

        self.mask_ratio = model_params.get('mask_ratio', 0.0)

    def compress_dims(self):
        """compress the hidden dims in the hidden layers
        Returns:
            _type_: _description_
        """
        return NotImplementedError()
    
    def encode(self):
        """encodes (compresses) an input x to latent z

        Returns:
            tensor: latent z
        """
        return NotImplementedError()

    def decode(self):
        """decodes (reconstructs) a latent z to output y

        Returns:
            tensor: latent z
        """
        return NotImplementedError()
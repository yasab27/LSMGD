"""
In this file we define classes related to the construction of a causal AE. This is a causal autoencoder where the encoder
half of the network consists of a Causal encoder object from from CausalCNN class and the decoder is a simple MLP. The
causal CNN feature extractor can be easily seperated from the remainder of the class to enable encoding. 
"""

__author__ = "Yasa Baig"

import torch
from models.CausalCNN import CausalCNNEncoder # Pull the encoder from the other file


class CausalAE(torch.nn.Module):
    """
    This network consists of a causal feature extractor adapated from [2] followed by a relatively small MLP decoder.
    
    Properties
    ----------
    encoder: CausalCNNEncoder
        The encoder half of the network which compresses our time series values to a low dimension.
        
    decoder: torch.nn.Sequential
        The decoder consisting of a MLP reconstructor.
    """
    
    def __init__(self, hp):
        """Initialize a new causal encoder.
        
        Parameters
        ----------
        hp: dict
            Dictionary containing the hyperparameters for both the encoder and decoder halfs of the network
        """
        super().__init__()
        
        self.hp = hp
        
        # Fix our CausalCNNEncoder based on [2]. This will in take a tensor of rank 3 with dimensions (B,C,L)
        # where B is the batch size (as usual), C is the number of input channels fed to the network. Physically,
        # this corresponds to how many variables are to be integrated (for ex. the price values for three different stocks)
        # and lastly L, the length of the time series. Since we are studying one dimensional reconstruction C = 1. 
        self.encoder = CausalCNNEncoder(
            in_channels = hp["in_channels"],
            channels = hp["channels"],
            depth = hp["depth"],
            reduced_size = hp["reduced_size"],
            out_channels = hp["out_channels"],
            kernel_size = hp["kernel_size"]
        )
        
        # Now define a fixed, hard coded reconstruction system.
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hp["out_channels"],25),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(25,50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50,75),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(75, hp["window_length"]) # Final dimension is encoded in the window_length
        )
        
    def forward(self, X):
        """Perform a forward pass through the network with training/test data X.
        
        Paramaters
        ----------
        X: tensor
            A rank 3 torch tensor of dimension (B,C,L) where B is the the batch size, C is the in_channels length (for us this
            will be generally only one 1), and L is the length. 
        
        Returns
        -------
        Y: tensor
            A rank 3 torch reconstructed from input after decoding. This matches the shape of the input and is reshaped manually.
        """
        
        code = self.encoder(X)
        code = code.squeeze() # Here we squeeze the last dimension of the tensor to be of size (B,C_out) so it can be fed
                              # into our network. This will have C_out = 3 by default. 
        Y = self.decoder(code)
        Y = Y.view((-1, 1, self.hp["window_length"])) # Hardcode reshaping to math input
        
        return Y, code
        
        
        
"""
In this file, we implement a simple framework for a fully convolutional variational autoencoder for compressing time series. Unlike a standard autoencoder, we essentially
regularize the latent space to ensure that it follows a normal distribution with "disentangled" feature representations by penalizing reconstruction error and the KL
divergence of the output and latent distributions. 
"""

import torch
import torch.nn
from torch.nn import functional as F

class SimpleVAE(torch.nn.Module):
    """Create a new simple 1D convolutional variational autoencoder. This is simple a to a regular autoencoder
       but we regularize the distribution to a normal. 
    """
    
    def __init__(self, hp):
        """
        Parameters
        ----------
        hp: dictionary
            Hyperparameters of the model architecture, most importantly the `embedding_dimension` for the size of the 
            hidden state. 
        """
        
        # Call parent model constructor and store hidden state variables.
        super().__init__()
        self.hp = hp
        
        
        # Initialize the convolutional feature extractor for the time series analysis. 
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels = hp["in"], out_channels = 16, kernel_size = hp["kernel_size"]),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size = hp["kernel_size"]),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size = hp["kernel_size"]),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size = hp["kernel_size"]),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size = hp["kernel_size"]),
            torch.nn.LeakyReLU(),
#             torch.nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = hp["kernel_size"]),
#             torch.nn.LeakyReLU(),
#             torch.nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = hp["kernel_size"]),
#             torch.nn.LeakyReLU(),
        )
        
        
        # Create branching hidden states for the embedding dimension, one for the mean and one for the standard
        # deviation. 
        self.mean_map = torch.nn.Sequential(
            torch.nn.Linear(160, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, hp["embedding_dimension"])
        )
        self.std_map = torch.nn.Sequential(
            torch.nn.Linear(160, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, hp["embedding_dimension"])
        )
        
        
        # Initiliaze the convolutional decoder module.
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(hp["embedding_dimension"], 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 160),
            torch.nn.LeakyReLU(),
        )

        
        self.decoder = torch.nn.Sequential(
#             torch.nn.ConvTranspose1d(in_channels = 32, out_channels = 32, kernel_size = hp["kernel_size"]),
#             torch.nn.LeakyReLU(),
#             torch.nn.ConvTranspose1d(in_channels = 32, out_channels = 32, kernel_size = hp["kernel_size"]),
#             torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(in_channels = 16, out_channels = 16, kernel_size = hp["kernel_size"]),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(in_channels = 16, out_channels = 16, kernel_size = hp["kernel_size"]),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(in_channels = 16, out_channels = 16, kernel_size = hp["kernel_size"]),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(in_channels = 16, out_channels = 16, kernel_size = hp["kernel_size"]),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(in_channels = 16, out_channels = hp["in"], kernel_size = hp["kernel_size"]),
        )
        
    def sample(self, mean, log_var):
        """Sample a given N(0,1) normal distribution given a mean and log of variance."""
        
        # First compute the variance from the log variance. 
        var = torch.exp(0.5*log_var)
        
        # Compute a scaled distribution
        eps = torch.randn_like(var)
        
        # Add the vectors
        z = mean + var*eps
        
        return z
        
    def forward(self, X):
        """Forward propogate through the model, return both the reconstruction and sampled mean and standard deviation
        for the system. 
        """
        
        # Now pass the information through the convolutional feature extracto
        pre_code = self.encoder(X)
                
        # Get the dimensionf othe precode
       
        
        # Reshape the tensor dimension for latent space sampling
        
        B, C, L = pre_code.shape[0], pre_code.shape[1], pre_code.shape[2]
        flattened = pre_code.view(B,C*L)
        
        
        # Now sample from the latent distribution for these points
        mu = self.mean_map(flattened)
        log_var = self.std_map(flattened)
        
        code = self.sample(mu, log_var)
        
        # Now pass the information through the decoder. Note we pass the last layer through a sigmoid
        # for the BCS loss
        post_code = self.linear2(code)
        
        
        X_hat = self.decoder( post_code.view(B, C, L)).squeeze()
                          
                          
        return X_hat, code, mu, log_var
    
    @staticmethod
    def vae_loss(x_hat, x, mu, log_var, alpha, gamma = 0):
        "Compute the sum of BCE and KL loss for the distribution."

        # Compute the reconstruction loss
        BCE = F.mse_loss(x_hat, x)

        # Compute the KL divergence of the distribution. 
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Normalize by the number of points in the batch and the dimensionality
        KLD /= (x.shape[0]*x.shape[1])
        
        SSL = F.mse_loss(x_hat[:,:,-1], x[:,:,-1])
        # ICL = F.mse_loss(x_hat[:,:,0], x[:,:,0])

        return BCE + alpha*KLD + gamma*SSL
        
    
    

        
        
        
        
    




    
    

        
        
        
        
    



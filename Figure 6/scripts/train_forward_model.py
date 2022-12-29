# Begin by importing critical dependencies for writing files and parsing through the local file system. 
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# We also setup torch for later usage.
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import trange
from torch.nn import functional as F

from sklearn.model_selection import train_test_split 

# Configure GPU if available
if torch.cuda.is_available():
    # device = "cuda:0"
    
    device = "cuda:0"
else:
    device = "cpu"

print(device)


#####################################################
# EDIT HYPERPARAMETERS HERE, THEY ARE NOT SUPPLIED  #
# BY CALLING SCRIPT EXCEPT FOR EXPERIMENT_NAME      #
# AND EM_DIM. UPDATE IN_DIM FOR SIZE OF COMMUNITY   #
#####################################################

# Specify hps to populate
hp = {
    "experiment_name": "NULL", 
    "data_number": 3, 
    "kernel_size": 3,
    "embedding_dimension": 8,
    "lr" : 1e-3,
    "weight_decay" : 0,
    "epochs": 5000,
    "batch_size": 10000,
    "alpha": 1e-4,
    "depth": 4,
    "in_dim": 3
}

#####################################################
#####################################################

# Populate all the hyperparameters
print(sys.argv )
script_name = sys.argv[0]
hp["experiment_name"] = sys.argv[1]
experiment_name = hp["experiment_name"]

em_dim = int(sys.argv[2])

import torch
import torch.nn
from torch.nn import functional as F

class EncoderMap(torch.nn.Module):
    
    def __init__(self, base_model, B, C, L):
        super().__init__()
        
        n = hp["in_dim"]
        layer_size = max(n*(n+1), 10)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear( (n*(n+1)) ,layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(layer_size,layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(layer_size,layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(layer_size,layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(layer_size,layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(layer_size, em_dim)
        )
        
        self.linear2 = base_model.linear2.eval()
        self.decoder = base_model.decoder.eval()
    
                
        # Get the dimensionf othe precode
        self.B = B
        self.C = C
        self.L = L
        
    def forward(self, X, ):
        
        B = X.shape[0]
        code = self.mlp(X)
        post_code = self.linear2(code)
        post_code = post_code.view(B,C,L)
        X_hat = self.decoder(post_code)
        
        return X_hat
    
class IVCurveDataSet(torch.utils.data.Dataset):
    
    def __init__(self, ivs, curves):
        
        self.X = ivs
        self.Y = curves
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        
        return self.X[idx, :], self.Y[idx, :, :]
    

#########################################################################
# EDIT HERE TO CHANGE THE DATA SOURCE. BY DEFAULT USES FIGURE DATASETS  #
#                                                                       #
#########################################################################

in_dim = hp["in_dim"]
file_name = f"data/param_space_{in_dim}_3.npy"
data = np.load(file_name)
X = data

file_name = f"data/param_space_{in_dim}_3_labels.npy"
data = np.load(file_name)
Y = data

#########################################################################
#########################################################################

print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,  test_size = 0.25, random_state = 42)

X_train = torch.Tensor(X_train).double().to(device)
Y_train = torch.Tensor(Y_train).double().to(device)
data_loader = DataLoader( IVCurveDataSet(Y_train, X_train), batch_size = 10000)

# Load in encoder
trained_model = torch.load(f"./saved_models/{in_dim}_FINAL_member_ablation_study_{em_dim}/model.pt")
X = torch.Tensor(X).to(device).double()


# Generate parameter size
pre_code = trained_model.encoder(X)
B, C, L = pre_code.shape

mlp_decoder = EncoderMap(trained_model,B, C, L ).to(device).double()

# from torch.optim.lr_scheduler import StepLR
def train_mlp_decoder(model, hp, data_loader, debug = False):

    
    # Store the losses per epoch
    running_losses = []
    decoder_norms = []
    encoder_norms = []
    
    # Configure optimizer and scheduler.
    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=hp["lr"], weight_decay= hp["weight_decay"])
    # scheduler = StepLR(optimizer, step_size = 1500, gamma = 0.1)
    
    # Outerloop will iterate through epochs. tqdm function trange provides progressbar
    for i in trange(hp["epochs"]):
        
        
        epoch_loss = 0 
        # Inner loop iterates through batches
        for iv, curves in data_loader:

            # Transfer the batch to the GPU
            iv = iv.to(device)
            curves = curves.to(device)

            if debug:
                print("BATCH SHAPE: ")
                print(batch.shape)

            # Zero gradient
            optimizer.zero_grad()

            # Perform forward pass

            recons = model(iv)

            # Uncomment to verify model prediction shape
            if debug:
                print("PREDI SHAPE: ")
                print(batch.shape)

            # Compute reconstruction loss
            batch_loss = F.mse_loss(recons, curves)
            
            if debug:
                print(batch_loss)

            # Compute gradient
            batch_loss.backward()

            # Take step
            optimizer.step()

            # Append to running epoch loss
            epoch_loss += batch_loss.item()

        # Keep running track of losses
        if i % 1000 == 0:
            print(f"Epoch [{i}]: " + str(epoch_loss))
            
            decoder_norm = 0
            for layer in model.decoder:
                if hasattr(layer, "weight"):
                    decoder_norm += layer.weight.norm()
            decoder_norms.append(decoder_norm)
            
            print("DECODER WEIGHTS: ", decoder_norm.item())
    
        running_losses.append(epoch_loss)

    return model, running_losses, decoder_norms

# Train model

trained_mlp_decoder , running_losses, decoder_norms = train_mlp_decoder(mlp_decoder, hp, data_loader)


# Save reults
plt.figure(figsize = (6,6))
plt.plot(torch.Tensor(running_losses).cpu())
plt.xlabel("Epochs")
plt.ylabel("Mean Square Error")
plt.savefig(f"./figures/{experiment_name}/loss.png")


trained_mlp_decoder = trained_mlp_decoder.eval()

torch.save(trained_mlp_decoder, f"./saved_models/{experiment_name}/model.pt")

print("finished!")
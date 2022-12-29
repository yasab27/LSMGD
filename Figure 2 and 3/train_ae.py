"""
Train a CausalAE model for a given dataset. 
"""

#######################################################################################################################
# 1. Loading Dependencies

__author__ = "Yasa Baig"

# We begin by importing critical dependencies for pytorch, numerical processing, plotting, and single cell analysis.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Training utilities
from torch.utils.data import DataLoader
from tqdm import trange 
from torch.optim.lr_scheduler import StepLR

# Lastly access system dependencies
import sys
import os

# Import our model
from models.CausalAE import CausalAE

# Define seaborn styling utilities for the plot creation. 
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context("notebook")


# If this is on GPU computation mode, switch the system to use CUDA over CPU for processing. Print which one is
# executing.
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
    
print(f"COMPUTING ON {device}")

#######################################################################################################################
# 2. Defining Training Functions

# Now we will define function for preprocessing the dataset.
def preprocess_data(file_path = "./data/env_isolate_curves.csv", batch_size = 0):
    """Given a file_path to a the file object, load in the dataset and prepare a data loader for autoencoding."""
        
    ###############################################################################
    #                                                                             #
    # CHANGE The data path here to edit which dataset to embed. By default        #
    # this will use the environmental isolate dataset.                            #
    #                                                                             #
    ###############################################################################
    data_path = "./data/env_isolate_curves.csv"
    
    X = pd.read_csv(data_path, header = None).to_numpy()
    
    # Min-Max normalize the entire dataset to the [0,1] range to improve training
    X = (X-X.min())/(X.max() - X.min()) 
    X = torch.Tensor(X)

    # Reshape to match the dimensions of the encoder. 
    X = X.reshape(( X.shape[0], 1, -1))
    
    # Generate DataLoader and return it
    if batch_size == 0:
        data_loader = DataLoader(X, X.shape[0])
    
    return X,  data_loader

def train(model, hp, data_loader, debug = False):
    
    running_losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay= hp["weight_decay"])
    
    criterion = torch.nn.MSELoss()
    
    for i in trange(hp["epochs"]):
        
        epoch_loss = 0
        
        for batch in data_loader:

            # Transfer the batch to the GPU
            batch = batch.to(device)

            if debug:
                print("BATCH SHAPE: ")
                print(batch.shape)

            # Zero gradient
            optimizer.zero_grad()

            # Perform forward pass
            pred, code = model(batch)

            if debug:
                print("PREDI SHAPE: ")
                print(pred.shape)

            batch_loss = criterion(pred,batch)
            
            if debug:
                print(batch_loss)

            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        # Keep running track of losses
        if i % 1000 == 0:
            print(f"Epoch [{i}]: " + str(epoch_loss))
            
            total_norm = 0

            for p in model.encoder.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            print("Encoder norm:", total_norm)
            
            total_norm = 0
            
            for p in model.decoder.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            print("Decoder norm:", total_norm)
            print("")
            
        running_losses.append(epoch_loss)
        

    return model, running_losses

#######################################################################################################################
# 3. Training Models

# Now we will train the autoencoder for a given dataset. 

# First we will need to set hyperparameters for our model training. Some of these will be passed as arguments to the 
# script. 
script_name, experiment_name, em_dim, depth = sys.argv
em_dim = int(em_dim)
depth = int(depth)

# Load the dataset
X, data_loader = preprocess_data() # Using default parameters

# Now we define a hyperparameter dictionary for usage in this model. These hyperparameters are fixed in advance. 

###############################################################################
#                                                                             #
# CHANGE The hyperparameters here path here to edit the structure of the      #
# network and the parameters of training.                                     #
#                                                                             #
###############################################################################
    
hp = {
    "in_channels" : 1, 
    "channels": 30, 
    "depth": depth, # Passed as a CLI argument above
    "reduced_size" : 30,
    "out_channels" : em_dim,  # Vary the embedding dimension of the dataset, normally pased as a CLI argument. 
    "kernel_size": 3,
    "window_length": X.shape[-1],
    "lr": 0.001, 
    "epochs": 200,
    "batch_size": -1, # Set to -1 to use the full dataset (maximum) batch size. 
    "weight_decay":0
}


# Define a model
model = CausalAE(hp)
model = model.to(device)

# Train the model
trained_model, losses = train(model, hp, data_loader )

#######################################################################################################################
# 4. Generate Figures and Embeddings

# First we will generate a figure for the perplexity and loss change throughout the course of training. 
plt.figure(figsize = (6,6))
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Mean Square Error")
plt.savefig(f"./figures/{experiment_name}/losses/ed_{em_dim}.png")

print("LOSS SAVED")

# 5. Save Models
torch.save(trained_model.state_dict(), f"./parameters/{experiment_name}/ed_{em_dim}.pt")

# Save embeddings
reconstructions, codes = trained_model(X.to(device))
codes = codes.detach().cpu().numpy()
reconstructions = reconstructions.detach().cpu().numpy()

# Normalize the latent codes
codes = (codes-codes.min())/(codes.max() - codes.min())
np.savetxt(f"./embeddings/{experiment_name}/ed_{em_dim}.csv", codes, delimiter = ",")

# Save 16 curves chosen at regular intervals
selected_curves = [100,200,300,400,
                   500,600,700,800,
                   900,1000,1100,1200,
                   1300,1400,1500,1600]

# Generate a subplot
fig, axs = plt.subplots(4,4, sharex = True, sharey = True, figsize = (15,15))
axs = axs.flatten()

for i in range(len(axs)):
    
    # Get the axis
    ax = axs[i]
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Grab a curve and its reconstruction
    curve = X[i,:].squeeze()
    recon = reconstructions[i,:].squeeze()
    
    ax.plot(curve)
    ax.plot(recon)
    ax.title.set_text(f"Curve {selected_curves[i]}")

plt.savefig(f"./figures/{experiment_name}/reconstructions/ed_{em_dim}.png")

#######################################################################################################################
# 5. Save hyperparameters
with open(f'./hyperparameters/{experiment_name}/hp_{experiment_name}_{em_dim}.txt', 'w') as f:
    print(hp, file=f)
    print(model, file = f)
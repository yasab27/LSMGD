 # Begin by importing critical dependencies for writing files and parsing through the local file system. 
import os
import sys

# Additionally import time dependencies for naming files
import time

# We begin by importing the essentials
import numpy as np
import matplotlib.pyplot as plt

# We also setup torch for later usage.
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import trange
from torch.nn import functional as F

from sklearn.model_selection import train_test_split 

print("Starting!")

# Configure GPU if available
if torch.cuda.is_available():
    # device = "cuda:0"
    
    device = "cuda:0"
else:
    device = "cpu"

print(device)

# Specify hps to populate
hp = {
    "experiment_name": "NULL", 
    "data_number": 0, 
    "kernel_size": 0,
    "embedding_dimension": 0,
    "lr" : 0,
    "weight_decay" : 0,
    "epochs": 0,
    "batch_size": 0,
    "alpha": 0,
    "depth": 0,
    "in":4
}

# Populate all the hyperparameters from the input arguments calling this script
print(sys.argv )
script_name = sys.argv[0]
hp["experiment_name"] = sys.argv[1]
experiment_name = hp["experiment_name"]

del hp["experiment_name"]

i = 2
for key in hp:
    value = float(sys.argv[i])


    if value.is_integer():
        value = int(value)
        
    hp[key] = value
    i += 1 
    
print(hp)  

number = hp["data_number"]

in_dim = hp["in"]

#########################################################################
# EDIT HERE TO CHANGE THE DATA SOURCE. BY DEFAULT USES FIGURE DATASETS  #
#                                                                       #
#########################################################################
file_name = f"./data/param_space_{in_dim}_3.npy"
X = np.load(file_name)
print(X.shape)
#########################################################################
#########################################################################


# Fixing the random state ensures that the same train/test split is generated 
# each time. 
X = (X - X.min())/(X.max() - X.min())
X_train, X_test = train_test_split(X, test_size = 0.25, random_state = 42)


from models.SimpleVAE import SimpleVAE

# Transfer the dataset to a tensor 
X_train = torch.Tensor(X_train).to(device).double()

# Dump it into a data loader
train_loader = DataLoader(X_train, batch_size = hp["batch_size"])

model = SimpleVAE(hp)
model.double()
model.to(device);

# from torch.optim.lr_scheduler import StepLR
def train(model, hp, data_loader, debug = False):
    """Train a given model for a specific type of hyperparamters and data_loader, train
    the model and then return the trained model as well as the running losses throughout 
    training.
    
    Parameters
    ----------
    model: torch.nn.Module
        An untrained model to be optimized.
    
    hp: dict
        A dictionary containing all of the hyperparameters for the system.
    
    data_loader: torch.utils.DataLoader
        A presetup dataloader containing the training data set for the set. 
    
    Returns
    -------
    model: torch.nn.Module
        The trained model after optimization.
        
    running_losses: list
        The loss for each epoch of training. 
    """
    
    # Store the losses per epoch
    running_losses = []
    
    # Configure optimizer and scheduler.
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay= hp["weight_decay"])
    # scheduler = StepLR(optimizer, step_size = 1500, gamma = 0.1)
    
    # Outerloop will iterate through epochs. tqdm function trange provides progressbar
    for i in trange(hp["epochs"]):
        
        
        epoch_loss = 0 
        # Inner loop iterates through batches
        for batch in data_loader:

            # Transfer the batch to the GPU
            batch = batch.to(device)

            if debug:
                print("BATCH SHAPE: ")
                print(batch.shape)

            # Zero gradient
            optimizer.zero_grad()

            # Perform forward pass
            pred, code, mu, log_var = model(batch)

            # Uncomment to verify model prediction shape
            if debug:
                print("PREDI SHAPE: ")
                print(batch.shape)

            # Compute reconstruction loss
            batch_loss = SimpleVAE.vae_loss(pred, batch, mu, log_var, hp["alpha"])

            if debug:
                print(batch_loss)

            # Compute gradient
            batch_loss.backward()

            # Take step
            optimizer.step()

            # Append to running epoch loss
            epoch_loss += batch_loss

        # Keep running track of losses
        if i % 250 == 0:
            print(f"Epoch [{i}]: " + str(epoch_loss))
    
        running_losses.append(epoch_loss)

    return model, running_losses


trained_model, running_losses = train(model, hp, train_loader, False)

### Analyze results

# Save the losses
plt.figure(figsize = (6,6))
plt.plot(torch.Tensor(running_losses).cpu())
plt.xlabel("Epochs")
plt.ylabel("Mean Square Error")
plt.savefig(f"./figures/{experiment_name}/loss.png")


trained_model = trained_model.eval()

torch.save(trained_model, f"./saved_models/{experiment_name}/model.pt")

print("finished!")
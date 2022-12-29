"""
This script bootstraps the execution of the multivqvae script on the GPU cluster. This operates by initializing a
shell file to sbatch and execute the required python file. 
"""

# Begin by importing critical dependencies for writing files and parsing through the local file system. 
import os
import sys

# Additionally import time dependencies for naming files
import time

# First we need to generate a directory called logs which can be used to store slum output files. If it already
# exists, do nothing, otherwise generate the folder. Similarly for figures and embeddings.
if not os.path.exists("logs/"):
    os.mkdir("logs/")
    
if not os.path.exists("figures/"):
    os.mkdir("figures/")
    
if not os.path.exists("embeddings/"):
    os.mkdir("embeddings/")
    
if not os.path.exists("parameters/"):
    os.mkdir("parameters/")
    
if not os.path.exists("saved_models/"):
    os.mkdir("saved_models/")

def gen_experiment_folders(experiment_name):
    # Generate a folder for this within logs, figures, and embeddigns
    if not os.path.exists(f"logs/{experiment_name}/"):
        os.mkdir(f"logs/{experiment_name}/")

    if not os.path.exists(f"embeddings/{experiment_name}/"):
        os.mkdir(f"embeddings/{experiment_name}/")

    if not os.path.exists(f"parameters/{experiment_name}/"):
        os.mkdir(f"parameters/{experiment_name}/")

    if not os.path.exists(f"figures/{experiment_name}/"):
        os.mkdir(f"figures/{experiment_name}/")
        
    if not os.path.exists(f"saved_models/{experiment_name}/"):
        os.mkdir(f"saved_models/{experiment_name}/")

    
# Now define a function to write the SBATCH script. We write this all as a single file for ease of reading and encapsulation.
def sbatch_run(command, job_name, experiment_name):
    """Takes a command (string) and executes it in an sbatch script on the machine."""
    
    # Initialize a temporary sh file for storing the shell script
    sh_file_name = ".temprun.sh"
    
    # Now initialize the script file contents, starting with a preamble basic scripting to provision memory and such
    sh_file = [
        "#!/bin/bash",
        "#SBATCH -p gpu-common --gres=gpu:1", # Specify the cs partition
        "#SBATCH --job-name=%s" % job_name, # Name job
        "#SBATCH --mem=5G", # Provision memorry
        f"#SBATCH -o logs/{experiment_name}/{job_name}.out", # Initialize location for the logs to output
        "source env/bin/activate", # Switch into virtual environment
        command # Execute the command
    ]
    
    # Now write the files
    with open(sh_file_name, 'w') as f:
        f.write('\n'.join(sh_file))
        
    print("Running %s" % job_name)
    
    # Now execute the sbatch command on the OS
    os.system('sbatch %s' % sh_file_name)
    
# Execute the script
data_number = 2

# Specify hps and execute job
hp = {
    "experiment_name": "NULL", 
    "data_number": 1, 
    "kernel_size": 3,
    "embedding_dimension": 8,
    "lr" : 1e-3,
    "weight_decay" : 0,
    "epochs": 100,
    "batch_size": 10000,
    "alpha": 1e-4,
    "depth": 4,
    "in_dim": 3
}


# Execute script
em_dims = [4,5,6,7,8,9]
for em_dim in em_dims:
    hp["embedding_dimension"] = em_dim
    hp["experiment_name"] = f"3_prediction_{em_dim}"
    exp_name = f"3_prediction_{em_dim}"
    gen_experiment_folders(hp["experiment_name"]);
    
    # Run sbatch command, dynamically append hyperparameters
    command_string = "python3 -u ./scripts/train_forward_model.py"
    command_string += f" {exp_name}"
    command_string += f" {em_dim}"

    sbatch_run(command = command_string, job_name = f"{em_dim}_ablation", experiment_name = hp["experiment_name"])
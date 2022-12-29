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
    
# Execute the script
depth = 5 # Complexity of encoder
em_dims = [2,3,5,10,15,20,25,30]

    
# Now we define this specific experiment's name. We want to store the logs and other information seperately
experiment_name = f"Example_Experiment"

# Generate a folder for this within logs, figures, and embeddigns
if not os.path.exists(f"logs/{experiment_name}/"):
    os.mkdir(f"logs/{experiment_name}/")
    
if not os.path.exists(f"embeddings/{experiment_name}/"):
    os.mkdir(f"embeddings/{experiment_name}/")
    
if not os.path.exists(f"parameters/{experiment_name}/"):
    os.mkdir(f"parameters/{experiment_name}/")
    
if not os.path.exists(f"figures/{experiment_name}/"):
    os.mkdir(f"figures/{experiment_name}/")
    os.mkdir(f"figures/{experiment_name}/losses/")
    os.mkdir(f"figures/{experiment_name}/reconstructions/")
    os.mkdir(f"figures/{experiment_name}/classification_results/")

if not os.path.exists(f"hyperparameters/{experiment_name}/"):
    os.mkdir(f"hyperparameters/{experiment_name}/")
    
# Now define a function to write the SBATCH script. We write this all as a single file for ease of reading and encapsulation.
def sbatch_run(command, job_name):
    """Takes a command (string) and executes it in an sbatch script on the machine."""
    
    # Initialize a temporary sh file for storing the shell script
    sh_file_name = ".temprun.sh"
    
    # Now initialize the script file contents, starting with a preamble basic scripting to provision memory and such
    sh_file = [
        "#!/bin/bash",
        "#SBATCH -p gpu-common --gres=gpu:1", # Specify the compute partition. NOTE THIS NEEDS BE TO BE SPECIFIC FOR EACH SYSTEM. 
        "#SBATCH --job-name=%s" % job_name, # Name job
        "#SBATCH --mem=25G", # Provision memorry
        f"#SBATCH -o logs/{experiment_name}/{job_name}.out", # Initialize location for the logs to output
        # "source env/bin/activate", # Switch into virtual environment
        command # Execute the command
    ]
    
    # Now write the files
    with open(sh_file_name, 'w') as f:
        f.write('\n'.join(sh_file))
         
    print("Running %s" % job_name)
    
    # Now execute the sbatch command on the OS
    os.system('sbatch %s' % sh_file_name)
    

# Run in parallel, embedding for each choice of embedding dimension. 
for em_dim in em_dims:
    sbatch_run(command = f"python3 -u ./train_ae.py {experiment_name} {em_dim} {depth}", job_name = f"{em_dim}_ed")
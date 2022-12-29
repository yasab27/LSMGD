In this folder, we provide scripts necessary for:

1) Generating low dimensional embeddings of all datasets 

2) Classifying embeddings and whole growth curves by either strain identity or antibiotic resistance phenotype.


#### Generating Low Dim Embedding

To generate low dimensional embeddings, use the the `train_ae.py` and `run_ae_experiment.py` script. Edit the `train_ae.py` script
to change which dataset to embed and modify the hyperparameters of the neural network. Initiate the `run_ae_experiment.py` script after editing `train_ae.py` to train multiple AEs with varying embedding dimensions but all other hyperparameters fixed (an embedding ablation study). It will be necessary to edit `run_ae_experiment.py` slurm commands to be consistent with a personal machine while `train_ae.py` should not have to be run independently. In `run_ae_experiment.py` all figures, embeddings, and hyperparameters from the model will be determined by user specified `experiment_name` folders. 

#### Run Classification of Embeddings

To run a classification experiments, specify the dataset of interest and labels in the `generate_ablation.py` script. To generate classification results, then run the `generate_ablation.py {EXPERIMENT_NAME}` where `EXPERIMENT_NAME` is a previously specified experiment name from the low dimensional embedding scripts. For antibiotic classification, a seperate script is provided for convenience which will generate classification curves for all four antibiotic label sets for a given embedding set all at once and is operated analogously (but there should never be a need to respecify the datasets/labels for that script). 

Results are stored in the `figures` folder for an experiment. We provide a sample experiment for the antibiotic classification already saved as the `antibiotic_final` embeddings. 



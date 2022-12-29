"""Generate a plot of classification accuracies vs embedding dimension for a given experiment."""


#######################################################################################################################
# 1. Loading Dependencies

__author__ = "Yasa Baig"

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import sys

# Import plotting libraries and change the style
import seaborn as sns
sns.set_style('white')
sns.set_context("paper")

# Import classifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# Import TQDM for plotting purposes
from tqdm import trange, tqdm

#######################################################################################################################
# 2. Define Helper Functions

def gen_plot(max_score, dimensions, embed_scores, title, ytitle, experiment_name, color = "r"):
    # Configure the plotting stlye
    plt.figure(figsize=(10,6))
    plt.rc('font', family='serif')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True,
        zorder = 20) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        right=False,      # ticks along the bottom edge are off
        left=True,         # ticks along the top edge are off
        labelleft=True) # labels along the bottom edge are off

    plt.ylim(0.0,1.05)
    plt.xlim(0,30)
    # Plot the data
    plt.plot([0, max(dimensions)+20],[max_score, max_score], "--",linewidth = 3, label = "Full Dimension Accuracy")
    plt.scatter(dimensions, embed_scores, c = color)
    plt.plot(dimensions, embed_scores, c = color, linewidth = 4)
    plt.legend(loc = "lower right")
    plt.xlabel("Embedding Dimension")
    plt.ylabel(ytitle)
    plt.title(title, pad = 10)
    plt.savefig(f"./figures/{experiment_name}/classification_results/{title}_{ytitle}.png")

def fit_clf(X_train,y_train, X_test, y_test, clf, parameters):
    """Given raw data, generate a training and testing set and then GridSearch over the dataset to 
    identify the best possible combinations for a classifer. Train a model over this, and then return 
    the best accuracy and the best model. 
    """
    # Now grid search over teh data and fit the best possible hyperparameters. Note we are using a threefold
    # cross validation procedure. 
    model_tuning = GridSearchCV(clf, param_grid=parameters,cv = 3, scoring='accuracy', verbose=1, n_jobs = -1)
    model_tuning.fit(X_train, y_train);
    clf_best = model_tuning.best_estimator_
    
    # Refit our best hyperparameter setup onto our training data in its totality one last time and print out
    # the average score on the training data. We anticipate this to be a large accuracy. 
    clf_best.fit(X_train,y_train)
    train_score = clf_best.score(X_train,y_train)
    test_score = clf_best.score(X_test,y_test)
    
    return clf_best, train_score, test_score

#######################################################################################################################
# 3. Generate results

# Get the experiment name
script_name, experiment_name = sys.argv


# Define hyperparameters to search over
parameters = {
    "estimator__C": [0.001, 0.01,0.1,1,10,25,50, 100],
    "estimator__kernel": ["rbf","linear",],
    "estimator__gamma":[ 0.1,0.01,1,10,25,50,100]
}


# Define embedding dimensions to search over
em_dims =  [2,3,5,10,20,25,30]

# Load in original dataset
for ant in ["CIP","GM","SAM", "SXT"]:
    X = pd.read_csv("./data/antibiotic_curves.csv", header = None).to_numpy()
    y = pd.read_csv(f"./data/{ant}_total_labels.csv", header = None).to_numpy().squeeze()

    # Fit a model to the original dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

    # Fit the model classifier on the dataset and find the optimal hyperparmeters
    clf = OneVsRestClassifier(SVC())
    clf, base_train_score, base_test_score = fit_clf(X_train, y_train, X_test, y_test, clf, parameters)
    print("BASE TRAIN SCORE: ",base_train_score )
    print("BASE TEST SCORE: ",base_test_score )


    # Git a model across all the dimensions
    clfs = []
    train_scores = []
    test_scores  = []

    # Loop though all of the data
    for em_dim in tqdm(em_dims):

        # First load the dataset 
        file_location =f"./embeddings/{experiment_name}/ed_{em_dim}.csv"
        X_dim = np.loadtxt(file_location, delimiter = ",")
        X_dim = (X_dim - X_dim.min())/(X_dim.max() - X_dim.min())
        print(X_dim.shape)

        # Generate train/test split using the labels from the original class
        X_train, X_test, y_train, y_test = train_test_split(X_dim, y, test_size = 0.25, random_state = 42)

        # Fit the model classifier on the dataset and find the optimal hyperparmeters
        clf = OneVsRestClassifier(SVC())
        clf, train_score, test_score = fit_clf(X_train, y_train, X_test, y_test, clf, parameters)
        print(f"TEST SCORE {em_dim}: ", test_score)
        # Store the performance metrics and classifiers in an array
        clfs.append(clf)
        train_scores.append(train_score)
        test_scores.append(test_score)

    # Generate the plots
    gen_plot(base_test_score, em_dims, test_scores, "Antibiotic Isolates", f"{ant} Testing Accuracy", experiment_name)
    gen_plot(base_train_score, em_dims, train_scores, "Antibiotic Isolates", f"{ant} Training Accuracy", experiment_name)
    
    final_data = np.array([em_dims, train_scores, test_scores]).T
    final_data = np.vstack([final_data, [0, base_train_score, base_test_score] ])

    np.savetxt(f"./figures/{experiment_name}/classification_results/{ant} Curves Final Data.csv", final_data, delimiter = ",")
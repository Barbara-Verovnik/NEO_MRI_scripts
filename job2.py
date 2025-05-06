#!/usr/bin/env python

#SBATCH --job-name=cv_job_atlas_64
#SBATCH --account=llonpp
#SBATCH --output=logs/output_%A_%a.out
#SBATCH --error=logs/error_%A_%a.err
#SBATCH --ntasks-per-node=72
#SBATCH --nodes=1
#SBATCH --time 02:30:00
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --mem=34000M
#SBATCH --export=ALL

import os
print(os.environ['PYTHONPATH']) 

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from util import RankMrMr, SelectFirstKFeatures
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
import sklearn
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, cohen_kappa_score, matthews_corrcoef, f1_score,accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score

# Load data
phenotype = 'age'  # see list below
method = "tangent"
atlas_dimension = 64 #Sizes: 64, 128, 256
print("Loading data.")

pheno_columns = {
    'age': "Age groups (1-60:79, 2-80:100+)",
    'sex': "Sex of subject at birth (1-M, 2-F)",
    'yrs_edu': "Education groups (1-7:15, 2-16:18, 3-19+)",
    'hand': "Handedness (1:R, 2:Mixed, 3:L)",
    'race': "Race (1:White, 2:Black, 3: Asian)",
    'lang': "Languages (1:1, 2:2, 3:3 and more)",
    'married': "Marital status (1:not married, 2:married, 3:divorced, widowed, separated)",
    'parity': "How many children do you have? (1:0, 2:1 or 2, 3:more than 2)",
    'employ': "Employed (1: not employed, 2: employed at least 6 months)",
}

path_to_excel = "demo_paper_data.xlsx"
df_meta = pd.read_excel(path_to_excel)

#Read connectome
X = np.load("Connectomes/XSTAN_" + str(atlas_dimension) + "_" + method + ".npy")
print("XSTAN_" + str(atlas_dimension) + "_" + method + ".npy has been loaded.")
y = df_meta[pheno_columns[phenotype]].to_numpy()

#Define model
sklearn.set_config(enable_metadata_routing=True)
# Necessary to pass sample weights through cross validation for hyperparameter tuning and and evaluation

mrmr = RankMrMr(verbose=True)
select = SelectFirstKFeatures(verbose=True)
nca =  NeighborhoodComponentsAnalysis(init="pca", max_iter=500, random_state=0)
knn = KNeighborsClassifier()

pipe = Pipeline([
    ('mrmr', mrmr),
    ('select', select),
    ("nca", nca),
    ("knn", knn)],
    memory='cache' # Enables storage caching to avoid recomputation
)

#Analysis 1

n_cv_splits = 5

param_grid = {
    "select__k":  [1,2,4,8,16,32,64,128,256,512,1024,2048,2080],
    "nca__n_components": [1, 2, 4, 8, 16, 32, 64],
    "knn__n_neighbors":  [1, 3, 5, 7, 9, 16+1, 32+1], # Odd numbers to help break ties
}
scoring = {
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'accuracy': make_scorer(accuracy_score), 
    'cohen_k': make_scorer(cohen_kappa_score),
    'mcc': make_scorer(matthews_corrcoef),
}

inner_cv = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=0)

gridsearch_1 = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring=scoring['f1_weighted'], # A sensible, multi-class capable metric that doesn't depend on class balance.
    cv=inner_cv,
    n_jobs=-1,
    verbose=4,
)

outer_cv = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=0)
print ("Starting calculations...")
cv_results = cross_validate(gridsearch_1, X, y, cv=outer_cv, scoring=scoring, verbose=4, n_jobs=1)

# Save the results
joblib.dump(cv_results, f"results/cv_results_{atlas_dimension}_{phenotype}.pkl")
print(f"cv_results saved to results/cv_results_{atlas_dimension}_{phenotype}.pkl")


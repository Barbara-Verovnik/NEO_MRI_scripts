import os
print(os.environ['PYTHONPATH'])
import sys

import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from util import RankMrMr, SelectFirstKFeatures
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
import sklearn
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, cohen_kappa_score, matthews_corrcoef, f1_score,accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict

# Get fold index
fold_idx = int(sys.argv[1])  # e.g. 0–4
print(f"Running outer CV fold {fold_idx}")

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

path_to_excel = "/data/leuven/369/vsc36935/SLURM/demo_paper_data.xlsx"
df_meta = pd.read_excel(path_to_excel)

#Read connectome
X = np.load("/data/leuven/369/vsc36935/SLURM/Connectomes/XSTAN_" + str(atlas_dimension) + "_" + method + ".npy")
print("XSTAN_" + str(atlas_dimension) + "_" + method + ".npy has been loaded.")
y = df_meta[pheno_columns[phenotype]].to_numpy()

#Define model
sklearn.set_config(enable_metadata_routing=True)
# Necessary to pass sample weights through cross validation for hyperparameter tuning and and evaluation

n_splits = 5

# Outer CV split
outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
splits = list(outer_cv.split(X, y))
train_idx, test_idx = splits[fold_idx]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Model pipeline
sklearn.set_config(enable_metadata_routing=True)

n_cv_splits = 5

pipe = Pipeline([
    ('mrmr', RankMrMr(verbose=True)),
    ('select', SelectFirstKFeatures(verbose=True)),
    ('nca', NeighborhoodComponentsAnalysis(init="pca", max_iter=500, random_state=0)),
    ('knn', KNeighborsClassifier())
], memory='cache')

# Hyperparameter grid
param_grid = {
    "select__k": [1,2,4],
    "nca__n_components": [1, 2, 4],
    "knn__n_neighbors": [1, 3, 5],
}

# Scoring dictionary (raw scorers for use inside GridSearchCV)
scoring = {
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'accuracy': make_scorer(accuracy_score),
    'cohen_k': make_scorer(cohen_kappa_score),
    'mcc': make_scorer(matthews_corrcoef),
}

# Inner CV
inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
gridsearch = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring['f1_weighted'], cv=inner_cv, n_jobs=1, verbose=2)

# Cross-validation on training set, scoring all metrics
cv_results = cross_validate(
    gridsearch,
    X_train,
    y_train,
    scoring=scoring,
    cv=inner_cv,
    return_estimator=True,
    n_jobs=-1,
    verbose=4,
)

# Save results
joblib.dump(cv_results, f"results/cv_results_{atlas_dimension}_{phenotype}_fold_{fold_idx}.pkl")
print(f"Fold {fold_idx} results saved.")

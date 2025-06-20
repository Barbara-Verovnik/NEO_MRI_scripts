import pandas as pd
from mrmr import mrmr_classif
from sklearn.base import BaseEstimator, TransformerMixin


class RankMrMr(BaseEstimator, TransformerMixin):

    def __init__(self, n_features=None, verbose=False):
        self.n_features = n_features
        self.verbose = verbose

    def fit(self, X, y=None):
        n_features = self.n_features
        if n_features is None:
            n_features = X.shape[-1]

        X = pd.DataFrame(X)
        y = pd.Series(y)

        self.support_ = mrmr_classif(X, y, K=n_features, show_progress=self.verbose)
        return self

    def transform(self, X, y=None):
        return X[:, self.support_]


class SelectFirstKFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, k=5, verbose=False):
        self.k = k
        self.verbose = verbose

    def fit(self, X, y=None):
        # No actual fitting required; just ensure k is valid
        if self.k > X.shape[1]:
            raise ValueError(
                f"k={self.k} is greater than the number of features ({X.shape[1]})"
            )
        if self.verbose:
            print(f"Keeping first {self.k} features")
        return self

    def transform(self, X):
        return X[:, : self.k]

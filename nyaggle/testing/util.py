from typing import Tuple
import pandas as pd
from sklearn.datasets import make_classification, make_regression


def make_classification_df(n_samples: int = 1024,
                           n_features: int = 20,
                           class_sep: float = 1.0,
                           feature_name: str = 'col_{}',
                           target_name: str = 'target',
                           random_state: int = 0,
                           id_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(n_samples=n_samples, n_features=n_features, class_sep=class_sep,
                               random_state=random_state)

    X = pd.DataFrame(X, columns=[feature_name.format(i) for i in range(n_features)])
    y = pd.Series(y, name=target_name)

    if id_column is not None:
        X[id_column] = range(n_samples)

    return X, y


def make_regression_df(n_samples: int = 1024,
                       n_features: int = 20,
                       feature_name: str = 'col_{}',
                       target_name: str = 'target',
                       random_state: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           random_state=random_state)

    X = pd.DataFrame(X, columns=[feature_name.format(i) for i in range(n_features)])
    y = pd.Series(y, name=target_name)

    return X, y

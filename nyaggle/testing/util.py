import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


def make_classification_df(n_samples: int = 1024,
                           n_num_features: int = 20,
                           n_cat_features: int = 0,
                           class_sep: float = 1.0,
                           n_classes: int = 2,
                           feature_name: str = 'col_{}',
                           target_name: str = 'target',
                           random_state: int = 0,
                           id_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    np.random.seed(random_state)
    X, y = make_classification(n_samples=n_samples, n_features=n_num_features, class_sep=class_sep,
                               random_state=random_state, n_classes=n_classes, n_informative=max(n_classes, 2))

    X = pd.DataFrame(X, columns=[feature_name.format(i) for i in range(n_num_features)])
    y = pd.Series(y, name=target_name)

    if id_column is not None:
        X[id_column] = range(n_samples)

    for i in range(n_cat_features):
        X['cat_{}'.format(i)] = \
            pd.Series(np.random.choice(['A', 'B', None], size=n_samples)).astype('category')

    return X, y


def make_regression_df(n_samples: int = 1024,
                       n_num_features: int = 20,
                       n_cat_features: int = 0,
                       feature_name: str = 'col_{}',
                       target_name: str = 'target',
                       random_state: int = 0,
                       id_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    np.random.seed(random_state)
    X, y = make_regression(n_samples=n_samples, n_features=n_num_features,
                           random_state=random_state)

    X = pd.DataFrame(X, columns=[feature_name.format(i) for i in range(n_num_features)])
    y = pd.Series(y, name=target_name)

    if id_column is not None:
        X[id_column] = range(n_samples)

    for i in range(n_cat_features):
        X['cat_{}'.format(i)] = \
            pd.Series(np.random.choice(['A', 'B', None], size=n_samples)).astype(str).astype('category')

    return X, y




@contextmanager
def get_temp_directory() -> str:
    path = None
    try:
        path = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
        yield path
    finally:
        if path:
            shutil.rmtree(path, ignore_errors=True)


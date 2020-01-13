from collections import namedtuple
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score

from nyaggle.validation.cross_validate import cross_validate

ADVResult = namedtuple('CVResult', ['auc', 'importance'])


def adversarial_validate(X_train: pd.DataFrame,
                         X_test: pd.DataFrame,
                         importance_type: str = 'gain',
                         estimator: Optional[BaseEstimator] = None,
                         single_fold: bool = True) -> ADVResult:
    """
    Perform adversarial validation between X_train and X_test.

    Args:
        X_train:
            Training data
        X_test:
            Test data
        importance_type:
            The type of feature importance calculated.
        estimator:
            The custom estimator. If None, LGBMClassifier is automatically used.
        single_fold:
            If True, validation score is calculated on 20% of data.
            If False, validation score is calculated by 5-fold cross-validation.

    Returns:
        Namedtuple with following members

        * auc:
            float, ROC AUC score of adversarial validation.
        * importance:
            pandas DataFrame, feature importance of adversarial model (order by importance)

    Example:
        >>> from sklearn.model_selection import train_test_split
        >>> from nyaggle.testing import make_regression_df
        >>> from nyaggle.validation import adversarial_validate

        >>> X, y = make_regression_df(n_samples=8)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
        >>> auc, importance = cross_validate(X_train, X_test)
        >>>
        >>> print(auc)
        0.51078231
        >>> importance.head()
        feature importance
        col_1   231.5827204
        col_5   207.1837266
        col_7   188.6920685
        col_4   174.5668498
        col_9   170.6438643
    """
    concat = pd.concat([X_train, X_test]).copy().reset_index(drop=True)
    y = np.array([1]*len(X_train) + [0]*len(X_test))

    if estimator is None:
        estimator = LGBMClassifier(n_estimators=10000, objective='binary', importance_type=importance_type,
                                   random_state=0)
    else:
        assert isinstance(estimator, (LGBMClassifier, CatBoostClassifier)), \
            'Only CatBoostClassifier or LGBMClassifier is allowed'

    if single_fold:
        nfolds_evaluate = 1
    else:
        nfolds_evaluate = None
    result = cross_validate(estimator, concat, y, None, cv=5, predict_proba=True,
                            eval_func=roc_auc_score, fit_params={'verbose': -1}, importance_type=importance_type,
                            nfolds_evaluate=nfolds_evaluate)

    importance = pd.concat(result.importance)
    importance = importance.groupby('feature')['importance'].mean().reset_index()
    importance.sort_values(by='importance', ascending=False, inplace=True)
    importance.reset_index(drop=True, inplace=True)

    return ADVResult(result.scores[-1], importance)

from collections import namedtuple
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from nyaggle.environment import requires_lightgbm
from nyaggle.util import is_instance
from nyaggle.validation.cross_validate import cross_validate
from nyaggle.validation.split import Take

ADVResult = namedtuple('ADVResult', ['auc', 'importance'])


def adversarial_validate(X_train: pd.DataFrame,
                         X_test: pd.DataFrame,
                         importance_type: str = 'gain',
                         estimator: Optional[BaseEstimator] = None,
                         cat_cols = None,
                         cv = None) -> ADVResult:
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
        cv:
            Cross validation split. If ``None``, the first fold out of 5 fold is used as validation.
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
        requires_lightgbm()
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(n_estimators=10000, objective='binary', importance_type=importance_type,
                                   random_state=0)
    else:
        assert is_instance(estimator, ('lightgbm.sklearn.LGBMModel', 'catboost.core.CatBoost')), \
            'Only CatBoostClassifier or LGBMClassifier is allowed'

    if cv is None:
        cv = Take(1, KFold(5, shuffle=True, random_state=0))

    fit_params = {'verbose': -1}
    if cat_cols:
        fit_params['categorical_feature'] = cat_cols

    result = cross_validate(estimator, concat, y, None, cv=cv, predict_proba=True,
                            eval_func=roc_auc_score, fit_params=fit_params, importance_type=importance_type)

    importance = pd.concat(result.importance)
    importance = importance.groupby('feature')['importance'].mean().reset_index()
    importance.sort_values(by='importance', ascending=False, inplace=True)
    importance.reset_index(drop=True, inplace=True)

    return ADVResult(result.scores[-1], importance)

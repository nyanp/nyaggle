from typing import Callable, Iterable, List, Union, Optional

import numpy as np
import pandas as pd
import sklearn.utils.multiclass as multiclass
from category_encoders.utils import convert_input, convert_input_vector
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from nyaggle.ensemble.common import EnsembleResult
from nyaggle.validation import cross_validate


def stacking(test_predictions: List[np.ndarray],
             oof_predictions: List[np.ndarray],
             y: pd.Series,
             estimator: Optional[BaseEstimator] = None,
             cv: Optional[Union[int, Iterable, BaseCrossValidator]] = None,
             groups: Optional[pd.Series] = None,
             type_of_target: str = 'auto',
             eval_func: Optional[Callable] = None) -> EnsembleResult:
    """
    Perform stacking on predictions.

    Args:
        test_predictions:
            List of predicted values on test data.
        oof_predictions:
            List of predicted values on out-of-fold training data.
        y:
            Target value
        estimator:
            Estimator used for the 2nd-level model.
            If ``None``, the default estimator (auto-tuned linear model) will be used.
        cv:
            int, cross-validation generator or an iterable which determines the cross-validation splitting strategy.

            - None, to use the default ``KFold(5, random_state=0, shuffle=True)``,
            - integer, to specify the number of folds in a ``(Stratified)KFold``,
            - CV splitter (the instance of ``BaseCrossValidator``),
            - An iterable yielding (train, test) splits as arrays of indices.
        groups:
            Group labels for the samples. Only used in conjunction with a “Group” cv instance (e.g., ``GroupKFold``).
        type_of_target:
            The type of target variable. If ``auto``, type is inferred by ``sklearn.utils.multiclass.type_of_target``.
            Otherwise, ``binary``, ``continuous``, or ``multiclass`` are supported.
        eval_func:
            Evaluation metric used for calculating result score. Used only if ``oof_predictions`` and ``y`` are given.
    Returns:
        Namedtuple with following members

        * test_prediction:
            numpy array, Average prediction on test data.
        * oof_prediction:
            numpy array, Average prediction on Out-of-Fold validation data. ``None`` if ``oof_predictions`` = ``None``.
        * score:
            float, Calculated score on Out-of-Fold data. ``None`` if ``eval_func`` is ``None``.
    """
    assert len(oof_predictions) == len(test_predictions), "Number of oof and test predictions should be same"

    def _stack(predictions):
        if predictions[0].ndim == 1:
            predictions = [p.reshape(len(p), -1) for p in predictions]
        return np.hstack(predictions)

    X_train = convert_input(_stack(oof_predictions))
    y = convert_input_vector(y, X_train.index)
    X_test = convert_input(_stack(test_predictions))

    assert len(X_train) == len(y)

    if type_of_target == 'auto':
        type_of_target = multiclass.type_of_target(y)

    if estimator is None:
        # if estimator is None, tuned linear estimator is used
        if type_of_target == 'continuous':
            estimator = Ridge(normalize=True, random_state=0)
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1, 10],
            }
        else:
            estimator = LogisticRegression(random_state=0)
            param_grid = {
                'penalty': ['l1', 'l2'],
                'C': [0.001, 0.01, 0.1, 1, 10],
            }
        grid_search = GridSearchCV(estimator, param_grid, cv=cv)
        grid_search.fit(X_train, y, groups=groups)
        estimator = grid_search.best_estimator_

    result = cross_validate(estimator, X_train, y, X_test, cv=cv, groups=groups, eval_func=eval_func)
    score = result.scores[-1] if result.scores else None

    return EnsembleResult(result.test_prediction, result.oof_prediction, score)

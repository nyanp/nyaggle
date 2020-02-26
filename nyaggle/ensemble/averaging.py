from collections import namedtuple
from typing import Callable, Iterable, List, Union, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

from nyaggle.ensemble.common import EnsembleResult


def averaging(test_predictions: List[np.ndarray],
              oof_predictions: Optional[List[np.ndarray]] = None,
              y: Optional[pd.Series] = None,
              weights: Optional[List[float]] = None,
              eval_func: Optional[Callable] = None,
              rank_averaging: bool = False) -> EnsembleResult:
    """
    Perform averaging on model predictions.

    Args:
        test_predictions:
            List of predicted values on test data.
        oof_predictions:
            List of predicted values on out-of-fold training data.
        y:
            Target value
        weights:
            Weights for each predictions
        eval_func:
            Evaluation metric used for calculating result score. Used only if ``oof_predictions`` and ``y`` are given.
        rank_averaging:
            If ``True``, predictions will be converted to rank before averaging.
    Returns:
        Namedtuple with following members

        * test_prediction:
            numpy array, Average prediction on test data.
        * oof_prediction:
            numpy array, Average prediction on Out-of-Fold validation data. ``None`` if ``oof_predictions`` = ``None``.
        * score:
            float, Calculated score on Out-of-Fold data. ``None`` if ``eval_func`` is ``None``.
    """
    if weights is None:
        weights = np.ones((len(test_predictions))) / len(test_predictions)

    if rank_averaging:
        test_predictions, oof_predictions = _to_rank(test_predictions, oof_predictions)

    def _weighted_average(predictions: List[np.ndarray], weights: List[float]):
        if len(predictions) != len(weights):
            raise ValueError('len(predictions) != len(weights)')
        average = np.zeros_like(predictions[0])

        for i, weight in enumerate(weights):
            if predictions[i].shape != average.shape:
                raise ValueError('predictions[{}].shape != predictions[0].shape'.format(i))
            average += predictions[i] * weight

        return average

    average_test = _weighted_average(test_predictions, weights)
    if oof_predictions is not None:
        average_oof = _weighted_average(oof_predictions, weights)
        score = eval_func(y, average_oof) if eval_func is not None else None
    else:
        average_oof = None
        score = None

    return EnsembleResult(average_test, average_oof, score)


def averaging_opt(test_predictions: List[np.ndarray],
                  oof_predictions: Optional[List[np.ndarray]],
                  y: Optional[pd.Series],
                  eval_func: Optional[Callable],
                  higher_is_better: bool,
                  weight_bounds: Tuple = (0, 1),
                  rank_averaging: bool = False) -> EnsembleResult:
    """
    Perform averaging with optimal weights using scipy.optimize

    Args:
        test_predictions:
            List of predicted values on test data.
        oof_predictions:
            List of predicted values on out-of-fold training data.
        y:
            Target value
        eval_func:
            Evaluation metric used for calculating result score. Used only if ``oof_predictions`` and ``y`` are given.
        higher_is_better:
            Determine the direction of optimize ``eval_func``.
        weight_bounds:
            Specify lower/upper bounds of each weight.
        rank_averaging:
            If ``True``, predictions will be converted to rank before averaging.
    Returns:
        Namedtuple with following members

        * test_prediction:
            numpy array, Average prediction on test data.
        * oof_prediction:
            numpy array, Average prediction on Out-of-Fold validation data. ``None`` if ``oof_predictions`` = ``None``.
        * score:
            float, Calculated score on Out-of-Fold data. ``None`` if ``eval_func`` is ``None``.
    """
    def _minimize(weights):
        prediction = np.zeros_like(oof_predictions[0])
        for weight, oof in zip(weights, oof_predictions):
            prediction += weight * oof
        oof_score = eval_func(y, prediction)

        return -oof_score if higher_is_better else oof_score

    weights = np.ones((len(test_predictions))) / len(test_predictions)

    if rank_averaging:
        test_predictions, oof_predictions = _to_rank(test_predictions, oof_predictions)

    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

    bounds = [weight_bounds] * len(test_predictions)

    result = minimize(_minimize, weights, method='SLSQP', constraints=cons, bounds=bounds)

    return averaging(test_predictions, oof_predictions, y, result['x'], eval_func)


def _to_rank(test_predictions: List[np.ndarray], oof_predictions: Optional[List[np.ndarray]]):
    if oof_predictions is not None:
        oof_predictions = [stats.rankdata(oof) / len(oof) for oof in oof_predictions]
    test_predictions = [stats.rankdata(test) / len(test) for test in test_predictions]

    return test_predictions, oof_predictions

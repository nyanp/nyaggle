from collections import namedtuple
from typing import Callable, Iterable, List, Union, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

from nyaggle.ensemble.common import EnsembleResult


def averaging(test_predictions: List[np.ndarray],
              oof_predictions: Optional[List[np.ndarray]] = None,
              y: Optional[pd.Series] = None,
              weights: Optional[List[float]] = None,
              eval_func: Optional[Callable] = None) -> EnsembleResult:
    if weights is None:
        weights = np.ones((len(test_predictions))) / len(test_predictions)

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


def rank_averaging(test_predictions: List[np.ndarray],
                   oof_predictions: Optional[List[np.ndarray]] = None,
                   y: Optional[pd.Series] = None,
                   weights: Optional[List[float]] = None,
                   eval_func: Optional[Callable] = None) -> EnsembleResult:
    def _to_rank(prediction: np.ndarray):
        return stats.rankdata(prediction) / len(prediction)

    if oof_predictions is not None:
        oof_rank_predictions = [_to_rank(oof) for oof in oof_predictions]
    else:
        oof_rank_predictions = None
    test_rank_predictions = [_to_rank(test) for test in test_predictions]

    return averaging(test_rank_predictions, oof_rank_predictions, y, weights, eval_func)
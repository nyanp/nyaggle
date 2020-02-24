from collections import namedtuple
from typing import Callable, Iterable, List, Union, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

from nyaggle.ensemble.common import EnsembleResult


def averaging(test_predictions: List[np.ndarray],
              oof_predictions: List[np.ndarray],
              y: pd.Series,
              weights: Optional[List[float]] = None,
              eval_func: Optional[Callable] = None) -> EnsembleResult:
    if weights is None:
        weights = np.ones((len(oof_predictions))) / len(oof_predictions)

    assert len(oof_predictions) == len(test_predictions), "Number of oof and test predictions should be same"
    assert len(oof_predictions) == len(weights), "Number of oof and weights should be same"

    average_oof = np.zeros_like(oof_predictions[0])
    average_test = np.zeros_like(test_predictions[0])

    for i, weight in enumerate(weights):
        if oof_predictions[i].shape != average_oof.shape:
            raise ValueError('oof_predictions[{}].shape != oof_predictions[0].shape'.format(i))
        if test_predictions[i].shape != average_test.shape:
            raise ValueError('test_predictions[{}].shape != test_predictions[0].shape'.format(i))

        average_oof += oof_predictions[i] * weight
        average_test += test_predictions[i] * weight

    score = eval_func(y, average_oof) if eval_func is not None else None

    return EnsembleResult(average_test, average_oof, score)


def rank_averaging(test_predictions: List[np.ndarray],
                   oof_predictions: List[np.ndarray],
                   y: pd.Series,
                   weights: Optional[List[float]] = None,
                   eval_func: Optional[Callable] = None) -> EnsembleResult:
    def _to_rank(prediction: np.ndarray):
        return stats.rankdata(prediction) / len(prediction)

    oof_rank_predictions = [_to_rank(oof) for oof in oof_predictions]
    test_rank_predictions = [_to_rank(test) for test in test_predictions]

    return averaging(test_rank_predictions, oof_rank_predictions, y, weights, eval_func)
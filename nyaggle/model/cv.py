import time
from collections import namedtuple
from logging import Logger, getLogger
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from category_encoders.utils import convert_input, convert_input_vector
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator
from lightgbm import LGBMClassifier, LGBMRegressor


CVResult = namedtuple('CVResult', ['predicted_oof', 'predicted_test', 'scores'])


def cv(model: Union[BaseEstimator, List[BaseEstimator]],
       X_train: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
       X_test: Union[pd.DataFrame, np.ndarray] = None,
       nfolds: int = 5, stratified: bool = False, seed: int = 42,
       predict_proba: bool = False, eval: Optional[Callable] = None, logger: Optional[Logger] = None,
       on_each_fold: Optional[Callable[[int, BaseEstimator, pd.DataFrame], None]] = None, **kw) -> CVResult:
    """
    Calculate Cross Validation

    Args:
        model:
            Model used in CV.
        X_train:
            Training data
        y:
            Target
        X_test:
            Test data (Optional). If specified, prediction on the test data is performed using ensemble of models.
        nfolds:
            Number of splits
        stratified:
            If true, use stratified K-Fold
        seed:
            Seed used by the random number generator in ``KFold``
        predict_proba:
            If true, call ``predict_proba`` instead of ``predict`` for calculating prediction for test data.
        eval:
            Function used for logging and returning scores
        logger:
            logger
        on_each_fold:
            called for each fold with (idx_fold, model, X_fold)
        kw:
            additional parameters passed to model.fit()

    Returns:
        Namedtuple with following members

        * predicted_oof:
            numpy array, shape (len(X_train),) Predicted value on Out-of-Fold validation data.
        * predicted_test:
            numpy array, shape (len(X_test),) Predicted value on test data. ``None`` if X_test is ``None``
        * scores:
            list of float, shape(nfolds+1) ``scores[i]`` denotes validation score in i-th fold.
            ``scores[-1]`` is overall score. `None` if eval is not specified

    Example:
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.metrics import mean_squared_error
        >>> from nyaggle.model import cv

        >>> X, y = make_regression(n_samples=8)
        >>> model = Ridge(alpha=1.0)
        >>> pred_oof, pred_test, scores = cv(model,
        >>>                                  X_train=X[:3, :],
        >>>                                  y=y[:3],
        >>>                                  X_test=X[3:, :],
        >>>                                  nfolds=3,
        >>>                                  eval=mean_squared_error)
        >>> print(pred_oof)
        [-101.1123267 ,   26.79300693,   17.72635528]
        >>> print(pred_test)
        [-10.65095894 -12.18909059 -23.09906427 -17.68360714 -20.08218267]
        >>> print(scores)
        [71912.80290003832, 15236.680239881942, 15472.822033121925, 34207.43505768073]
    """
    X_train = convert_input(X_train)
    y = convert_input_vector(y, X_train.index)
    if X_test is not None:
        X_test = convert_input(X_test)

    if stratified:
        folds = StratifiedKFold(n_splits=nfolds, random_state=seed)
    else:
        folds = KFold(n_splits=nfolds, random_state=seed)

    if not isinstance(model, list):
        model = [model] * nfolds

    assert len(model) == nfolds

    if logger is None:
        logger = getLogger(__name__)

    def _predict(model: BaseEstimator, x: pd.DataFrame, predict_proba: bool):
        if predict_proba:
            return model.predict_proba(x)[:, 1]
        else:
            return model.predict(x)

    oof = np.zeros(len(X_train))
    if X_test is not None:
        test = np.zeros((len(X_test), nfolds))

    scores = []
    eta_all = []

    for n, (train_idx, valid_idx) in enumerate(folds.split(X_train, y)):
        start_time = time.time()

        train_x, train_y = X_train.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = X_train.iloc[valid_idx], y.iloc[valid_idx]

        if isinstance(model[n], (LGBMRegressor, LGBMClassifier)):
            model[n].fit(train_x, train_y, eval_set=[(valid_x, valid_y)], early_stopping_rounds=100, **kw)
        else:
            model[n].fit(train_x, train_y, **kw)

        oof[valid_idx] = _predict(model[n], valid_x, predict_proba)

        if X_test is not None:
            test[:, n] = _predict(model[n], X_test, predict_proba)

        if on_each_fold is not None:
            on_each_fold(n, model[n], train_x)

        if eval is not None:
            score = eval(valid_y, oof[valid_idx])
            scores.append(score)
            logger.info('Fold {} score: {}'.format(n, score))

        elapsed = time.time() - start_time
        eta_all.append(elapsed)
        logger.debug('{:.3f} sec / fold'.format(elapsed))

    if eval is not None:
        score = eval(y, oof)
        scores.append(score)
        logger.info('Overall score: {}'.format(score))

    if X_test is not None:
        predicted = np.mean(test, axis=1)
    else:
        predicted = None

    return CVResult(oof, predicted, scores)

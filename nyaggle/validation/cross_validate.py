import copy
import time
from collections import namedtuple
from logging import Logger, getLogger
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from category_encoders.utils import convert_input, convert_input_vector
from catboost import CatBoost
from lightgbm import LGBMModel
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold

CVResult = namedtuple('CVResult', ['predicted_oof', 'predicted_test', 'scores', 'importance'])


def cross_validate(estimator: Union[BaseEstimator, List[BaseEstimator]],
                   X_train: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
                   X_test: Union[pd.DataFrame, np.ndarray] = None,
                   nfolds: int = 5, stratified: bool = False, seed: int = 42,
                   predict_proba: bool = False, eval: Optional[Callable] = None, logger: Optional[Logger] = None,
                   on_each_fold: Optional[Callable[[int, BaseEstimator, pd.DataFrame, pd.Series], None]] = None,
                   fit_params: Optional[Dict] = None,
                   importance_type: str = 'gain') -> CVResult:
    """
    Evaluate metrics by cross-validation. It also records out-of-fold prediction and test prediction.

    Args:
        estimator:
            The object to be used in cross-validation. For list inputs, ``estimator[i]`` is trained on i-th fold.
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
            called for each fold with (idx_fold, model, X_fold, y_fold)
        fit_params:
            Parameters passed to the fit method of the estimator

    Returns:
        Namedtuple with following members

        * predicted_oof:
            numpy array, shape (len(X_train),) Predicted value on Out-of-Fold validation data.
        * predicted_test:
            numpy array, shape (len(X_test),) Predicted value on test data. ``None`` if X_test is ``None``.
        * scores:
            list of float, shape(nfolds+1) ``scores[i]`` denotes validation score in i-th fold.
            ``scores[-1]`` is overall score. `None` if eval is not specified.
        * importance:
            list of pandas DataFrame, shape(nfolds,) ``importance[i]`` denotes feature importance in i-th fold.
            If estimator is not GBDT, empty array is returned.

    Example:
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.metrics import mean_squared_error
        >>> from nyaggle.validation import cross_validate

        >>> X, y = make_regression(n_samples=8)
        >>> model = Ridge(alpha=1.0)
        >>> pred_oof, pred_test, scores, _ = cross_validate(model,
        >>>                                                 X_train=X[:3, :],
        >>>                                                 y=y[:3],
        >>>                                                 X_test=X[3:, :],
        >>>                                                 nfolds=3,
        >>>                                                 eval=mean_squared_error)
        >>> print(pred_oof)
        [-101.1123267 ,   26.79300693,   17.72635528]
        >>> print(pred_test)
        [-10.65095894 -12.18909059 -23.09906427 -17.68360714 -20.08218267]
        >>> print(scores)
        [71912.80290003832, 15236.680239881942, 15472.822033121925, 34207.43505768073]
    """
    if isinstance(estimator, list):
        assert len(estimator) == nfolds, "Number of estimators should be same to nfolds."

    X_train = convert_input(X_train)
    y = convert_input_vector(y, X_train.index)
    if X_test is not None:
        X_test = convert_input(X_test)

    if stratified:
        folds = StratifiedKFold(n_splits=nfolds, random_state=seed)
    else:
        folds = KFold(n_splits=nfolds, random_state=seed)

    if not isinstance(estimator, list):
        estimator = [estimator] * nfolds

    assert len(estimator) == nfolds

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
    importance = []

    for n, (train_idx, valid_idx) in enumerate(folds.split(X_train, y)):
        start_time = time.time()

        train_x, train_y = X_train.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = X_train.iloc[valid_idx], y.iloc[valid_idx]

        if isinstance(estimator[n], LGBMModel):
            if fit_params is None:
                fit_params_fold = {}
            else:
                fit_params_fold = copy.copy(fit_params)
            if 'eval_set' not in fit_params_fold:
                fit_params_fold['eval_set'] = [(valid_x, valid_y)]
            if 'early_stopping_rounds' not in fit_params_fold:
                fit_params_fold['early_stopping_rounds'] = 100

            estimator[n].fit(train_x, train_y, **fit_params_fold)
        elif fit_params is not None:
            estimator[n].fit(train_x, train_y, **fit_params)
        else:
            estimator[n].fit(train_x, train_y)

        oof[valid_idx] = _predict(estimator[n], valid_x, predict_proba)

        if X_test is not None:
            test[:, n] = _predict(estimator[n], X_test, predict_proba)

        if on_each_fold is not None:
            on_each_fold(n, estimator[n], train_x, train_y)

        if isinstance(estimator[n], (LGBMModel, CatBoost)):
            importance.append(_get_gbdt_importance(estimator[n], list(X_train.columns), importance_type))

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

    return CVResult(oof, predicted, scores, importance)


def _get_gbdt_importance(gbdt_model: Union[CatBoost, LGBMModel], features: List[str],
                         importance_type: str) -> pd.DataFrame:
    df = pd.DataFrame()

    df['feature'] = features

    if isinstance(gbdt_model, CatBoost):
        df['importance'] = gbdt_model.get_feature_importance()
    elif isinstance(gbdt_model, LGBMModel):
        df['importance'] = gbdt_model.booster_.feature_importance(importance_type=importance_type)

    return df

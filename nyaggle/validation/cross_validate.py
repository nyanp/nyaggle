import copy
import time
import warnings
from collections import namedtuple
from logging import Logger, getLogger
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import sklearn.utils.multiclass as multiclass
from category_encoders.utils import convert_input, convert_input_vector
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from nyaggle.util.traits import is_gbdt_instance
from nyaggle.validation.split import check_cv, extract_source_of_time_series_split


CVResult = namedtuple('CVResult', ['oof_prediction', 'test_prediction', 'scores', 'importance'])


def cross_validate(estimator: Union[BaseEstimator, List[BaseEstimator]],
                   X_train: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
                   X_test: Union[pd.DataFrame, np.ndarray] = None,
                   cv: Optional[Union[int, Iterable, BaseCrossValidator]] = None,
                   groups: Optional[pd.Series] = None,
                   eval_func: Optional[Callable] = None, logger: Optional[Logger] = None,
                   on_each_fold: Optional[Callable[[int, BaseEstimator, pd.DataFrame, pd.Series], None]] = None,
                   fit_params: Optional[Union[Dict[str, Any], Callable]] = None,
                   importance_type: str = 'gain',
                   early_stopping: bool = True,
                   type_of_target: str = 'auto') -> CVResult:
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
        cv:
            int, cross-validation generator or an iterable which determines the cross-validation splitting strategy.

            - None, to use the default ``KFold(5, random_state=0, shuffle=True)``,
            - integer, to specify the number of folds in a ``(Stratified)KFold``,
            - CV splitter (the instance of ``BaseCrossValidator``),
            - An iterable yielding (train, test) splits as arrays of indices.
        groups:
            Group labels for the samples. Only used in conjunction with a “Group” cv instance (e.g., ``GroupKFold``).
        eval_func:
            Function used for logging and returning scores
        logger:
            logger
        on_each_fold:
            called for each fold with (idx_fold, model, X_fold, y_fold)
        fit_params:
            Parameters passed to the fit method of the estimator
        importance_type:
            The type of feature importance to be used to calculate result.
            Used only in ``LGBMClassifier`` and ``LGBMRegressor``.
        early_stopping:
            If ``True``, ``eval_set`` will be added to ``fit_params`` for each fold.
            ``early_stopping_rounds = 100`` will also be appended to fit_params if it does not already have one.
        type_of_target:
            The type of target variable. If ``auto``, type is inferred by ``sklearn.utils.multiclass.type_of_target``.
            Otherwise, ``binary``, ``continuous``, or ``multiclass`` are supported.
    Returns:
        Namedtuple with following members

        * oof_prediction (numpy array, shape (len(X_train),)):
            The predicted value on put-of-Fold validation data.
        * test_prediction (numpy array, hape (len(X_test),)):
            The predicted value on test data. ``None`` if X_test is ``None``.
        * scores (list of float, shape (nfolds+1,)):
            ``scores[i]`` denotes validation score in i-th fold.
            ``scores[-1]`` is the overall score. `None` if eval is not specified.
        * importance (list of pandas DataFrame, shape (nfolds,)):
            ``importance[i]`` denotes feature importance in i-th fold model.
            If the estimator is not GBDT, empty array is returned.

    Example:
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.metrics import mean_squared_error
        >>> from nyaggle.validation import cross_validate

        >>> X, y = make_regression(n_samples=8)
        >>> model = Ridge(alpha=1.0)
        >>> pred_oof, pred_test, scores, _ = \
        >>>     cross_validate(model,
        >>>                    X_train=X[:3, :],
        >>>                    y=y[:3],
        >>>                    X_test=X[3:, :],
        >>>                    cv=3,
        >>>                    eval_func=mean_squared_error)
        >>> print(pred_oof)
        [-101.1123267 ,   26.79300693,   17.72635528]
        >>> print(pred_test)
        [-10.65095894 -12.18909059 -23.09906427 -17.68360714 -20.08218267]
        >>> print(scores)
        [71912.80290003832, 15236.680239881942, 15472.822033121925, 34207.43505768073]
    """
    cv = check_cv(cv, y)
    n_output_cols = 1
    if type_of_target == 'auto':
        type_of_target = multiclass.type_of_target(y)
    if type_of_target == 'multiclass':
        n_output_cols = y.nunique(dropna=True)

    if isinstance(estimator, list):
        assert len(estimator) == cv.get_n_splits(), "Number of estimators should be same to nfolds."

    X_train = convert_input(X_train)
    y = convert_input_vector(y, X_train.index)
    if X_test is not None:
        X_test = convert_input(X_test)

    time_series_column = extract_source_of_time_series_split(cv) if cv else None

    if not isinstance(estimator, list):
        estimator = [estimator] * cv.get_n_splits()

    assert len(estimator) == cv.get_n_splits()

    if logger is None:
        logger = getLogger(__name__)

    def _predict(model: BaseEstimator, x: pd.DataFrame, _type_of_target: str):
        if _type_of_target in ('binary', 'multiclass'):
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)
            elif hasattr(model, "decision_function"):
                warnings.warn('Since {} does not have predict_proba method, '
                              'decision_function is used for the prediction instead.'.format(type(model)))
                proba = model.decision_function(x)
            else:
                raise RuntimeError('Estimator in classification problem should have '
                                   'either predict_proba or decision_function')
            if proba.ndim == 1:
                return proba
            else:
                return proba[:, 1] if proba.shape[1] == 2 else proba
        else:
            return model.predict(x)

    oof = np.zeros((len(X_train), n_output_cols)) if n_output_cols > 1 else np.zeros(len(X_train))
    evaluated = np.full(len(X_train), False)
    test = None
    if X_test is not None:
        test = np.zeros((len(X_test), n_output_cols)) if n_output_cols > 1 else np.zeros(len(X_test))

    if time_series_column is not None:
        X_test = X_test.drop(columns=time_series_column)

    scores = []
    eta_all = []
    importance = []

    for n, (train_idx, valid_idx) in enumerate(cv.split(X_train, y, groups)):
        start_time = time.time()

        train_x, train_y = X_train.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = X_train.iloc[valid_idx], y.iloc[valid_idx]

        if time_series_column is not None:
            train_x = train_x.drop(columns=time_series_column)
            valid_x = valid_x.drop(columns=time_series_column)

        if fit_params is None:
            fit_params_fold = {}
        elif callable(fit_params):
            fit_params_fold = fit_params(n, train_idx, valid_idx)
        else:
            fit_params_fold = copy.copy(fit_params)

        if is_gbdt_instance(estimator[n], ('lgbm', 'cat', 'xgb')):
            if early_stopping:
                if 'eval_set' not in fit_params_fold:
                    fit_params_fold['eval_set'] = [(valid_x, valid_y)]
                if 'early_stopping_rounds' not in fit_params_fold:
                    fit_params_fold['early_stopping_rounds'] = 100

            estimator[n].fit(train_x, train_y, **fit_params_fold)
        else:
            estimator[n].fit(train_x, train_y, **fit_params_fold)

        oof[valid_idx] = _predict(estimator[n], valid_x, type_of_target)
        evaluated[valid_idx] = True

        if X_test is not None:
            test += _predict(estimator[n], X_test, type_of_target)

        if on_each_fold is not None:
            on_each_fold(n, estimator[n], train_x, train_y)

        if is_gbdt_instance(estimator[n], ('lgbm', 'cat', 'xgb')):
            importance.append(_get_gbdt_importance(estimator[n], list(train_x.columns), importance_type))

        if eval_func is not None:
            score = eval_func(valid_y, oof[valid_idx])
            scores.append(score)
            logger.info('Fold {} score: {}'.format(n, score))

        elapsed = time.time() - start_time
        eta_all.append(elapsed)
        logger.debug('{:.3f} sec / fold'.format(elapsed))

    if eval_func is not None:
        score = eval_func(y.loc[evaluated], oof[evaluated])
        scores.append(score)
        logger.info('Overall score: {}'.format(score))

    if X_test is not None:
        predicted = test / cv.get_n_splits(X_train, y, groups)
    else:
        predicted = None

    return CVResult(oof, predicted, scores, importance)


def _get_gbdt_importance(gbdt_model: BaseEstimator, features: List[str],
                         importance_type: str) -> pd.DataFrame:
    df = pd.DataFrame()

    df['feature'] = features

    if is_gbdt_instance(gbdt_model, 'cat'):
        df['importance'] = gbdt_model.get_feature_importance()
    elif is_gbdt_instance(gbdt_model, 'xgb'):
        df['importance'] = gbdt_model.feature_importances_
    elif is_gbdt_instance(gbdt_model, 'lgbm'):
        df['importance'] = gbdt_model.booster_.feature_importance(importance_type=importance_type)

    return df

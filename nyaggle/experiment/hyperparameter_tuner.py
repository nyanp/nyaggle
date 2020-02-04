import copy
from typing import Dict, Iterable, Optional, Union

import pandas as pd
import optuna.integration.lightgbm as optuna_lgb
import sklearn.utils.multiclass as multiclass
from sklearn.model_selection import BaseCrossValidator

from nyaggle.validation.split import check_cv


def find_best_lgbm_parameter(base_param: Dict, X: pd.DataFrame, y: pd.Series,
                             cv: Optional[Union[int, Iterable, BaseCrossValidator]] = None,
                             groups: Optional[pd.Series] = None,
                             time_budget: Optional[int] = None,
                             type_of_target: str = 'auto') -> Dict:
    """
    Search hyperparameter for lightgbm using optuna.

    Args:
        base_param:
            Base parameters passed to lgb.train.
        X:
            Training data.
        y:
            Target
        cv:
            int, cross-validation generator or an iterable which determines the cross-validation splitting strategy.
        groups:
            Group labels for the samples. Only used in conjunction with a “Group” cv instance (e.g., ``GroupKFold``).
        time_budget:
            Time budget for tuning (in seconds).
        type_of_target:
            The type of target variable. If ``auto``, type is inferred by ``sklearn.utils.multiclass.type_of_target``.
            Otherwise, ``binary``, ``continuous``, or ``multiclass`` are supported.

    Returns:
        The best parameters found
    """
    cv = check_cv(cv, y)

    if type_of_target == 'auto':
        type_of_target = multiclass.type_of_target(y)

    train_index, test_index = next(cv.split(X, y, groups))

    dtrain = optuna_lgb.Dataset(X.iloc[train_index], y.iloc[train_index])
    dvalid = optuna_lgb.Dataset(X.iloc[test_index], y.iloc[test_index])

    params = copy.deepcopy(base_param)
    if 'early_stopping_rounds' not in params:
        params['early_stopping_rounds'] = 100

    if not any([p in params for p in ('num_iterations', 'num_iteration',
                                      'num_trees', 'num_tree',
                                      'num_rounds', 'num_round')]):
        params['num_iterations'] = params.get('n_estimators', 10000)

    if 'objective' not in params:
        tot_to_objective = {
            'binary': 'binary',
            'continuous': 'regression',
            'multiclass': 'multiclass'
        }
        params['objective'] = tot_to_objective[type_of_target]

    if 'metric' not in params and 'objective' in params:
        if params['objective'] in ['regression', 'regression_l2', 'l2', 'mean_squared_error', 'mse', 'l2_root',
                                   'root_mean_squared_error', 'rmse']:
            params['metric'] = 'l2'
        if params['objective'] in ['regression_l1', 'l1', 'mean_absolute_error', 'mae']:
            params['metric'] = 'l1'
        if params['objective'] in ['binary']:
            params['metric'] = 'binary_logloss'
        if params['objective'] in ['multiclass']:
            params['metric'] = 'multi_logloss'

    if not any([p in params for p in ('verbose', 'verbosity')]):
        params['verbosity'] = -1

    best_params, tuning_history = dict(), list()
    optuna_lgb.train(params, dtrain, valid_sets=[dvalid], verbose_eval=0,
                     best_params=best_params, tuning_history=tuning_history, time_budget=time_budget)

    result_param = copy.deepcopy(base_param)
    result_param.update(best_params)
    return result_param

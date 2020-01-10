from more_itertools import first_true
from typing import Dict, List, Union

from nyaggle.hyper_parameters.catboost import parameters as params_cat
from nyaggle.hyper_parameters.lightgbm import parameters as params_lgb
from nyaggle.hyper_parameters.xgboost import parameters as params_xgb


def _get_hyperparam_byname(param_table: List[Dict], name: str, with_metadata: bool):
    found = first_true(param_table, pred=lambda x: x['name'] == name)
    if found is None:
        raise RuntimeError('Hyperparameter {} not found.'.format(name))

    if with_metadata:
        return found
    else:
        return found['parameters']


def _return(parameter: Union[List[Dict], Dict], with_metadata: bool) -> Union[List[Dict], Dict]:
    if with_metadata:
        return parameter

    if isinstance(parameter, list):
        return [p['parameters'] for p in parameter]
    else:
        return parameter['parameters']


def _get_table(gbdt_type: str = 'lgb'):
    if gbdt_type == 'lgb':
        return params_lgb
    elif gbdt_type == 'cat':
        return params_cat
    elif gbdt_type == 'xgb':
        return params_xgb
    raise ValueError('gbdt type should be one of (lgb, cat, xgb)')


def list_hyperparams(gbdt_type: str = 'lgb', with_metadata: bool = False) -> List[Dict]:
    """
    List all hyperparameters
    Args:
        gbdt_type:
            The type of gbdt library. ``lgb``, ``cat``, ``xgb`` can be used.
        with_metadata:
            When set to True, parameters are wrapped by metadata dictionary which contains information about
            source URL, competition name etc.
    Returns:
        A list of hyper-parameters used in Kaggle gold medal solutions
    """
    return _return(_get_table(gbdt_type), with_metadata)


def get_hyperparam_byname(name: str, gbdt_type: str = 'lgb', with_metadata: bool = False) -> Dict:
    """
    Get a hyperparameter by parameter name
    Args:
        name:
            The name of parameter (e.g. "ieee-2019-10th").
        gbdt_type:
            The type of gbdt library. ``lgb``, ``cat``, ``xgb`` can be used.
        with_metadata:
            When set to True, parameters are wrapped by metadata dictionary which contains information about
            source URL, competition name etc.
    Returns:
        A hyperparameter dictionary.
    """
    param_table = _get_table(gbdt_type)
    found = first_true(param_table, pred=lambda x: x['name'] == name)
    if found is None:
        raise RuntimeError('Hyperparameter {} not found.'.format(name))

    return _return(found, with_metadata)

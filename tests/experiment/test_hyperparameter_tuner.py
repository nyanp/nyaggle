import pandas as pd
from sklearn import datasets

from nyaggle.experiment.hyperparameter_tuner import find_best_lgbm_parameter


def _check_parameter_tunes(params, x, y):
    best_params = find_best_lgbm_parameter(params, x, y)
    # lightgbm_tuner tuned params
    tuned_params = {
        'num_leaves', 'feature_fraction', 'bagging_fraction', 'bagging_freq',
        'lambda_l1', 'lambda_l2', 'min_child_samples'
    }
    intersection = set(best_params.keys()) & tuned_params
    assert intersection == tuned_params


def test_regression_problem_parameter_tunes():
    x, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
    }
    _check_parameter_tunes(params, x, y)


def test_binary_classification_parameter_tunes():
    dataset = datasets.load_breast_cancer()
    x = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
    }
    _check_parameter_tunes(params, x, y)


def test_multi_classification_parameter_tunes():
    dataset = datasets.load_wine()
    x = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target)
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'verbosity': -1,
    }
    _check_parameter_tunes(params, x, y)

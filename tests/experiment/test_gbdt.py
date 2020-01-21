import json
import os
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from nyaggle.experiment import experiment_gbdt, find_best_lgbm_parameter
from nyaggle.testing import make_classification_df, make_regression_df, get_temp_directory


def _check_file_exists(directory, files):
    for f in files:
        assert os.path.exists(os.path.join(directory, f)), 'File not found: {}'.format(f)


def test_experiment_lgb_classifier():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, X_test, temp_path, eval_func=roc_auc_score)

        assert roc_auc_score(y_train, result.oof_prediction) >= 0.85
        assert roc_auc_score(y_test, result.test_prediction) >= 0.85

        _check_file_exists(temp_path, ('oof_prediction.npy', 'test_prediction.npy', 'metrics.txt'))


def test_experiment_lgb_regressor():
    X, y = make_regression_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                              random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'regression',
        'max_depth': 8
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, X_test, temp_path)

        assert mean_squared_error(y_train, result.oof_prediction) == result.metrics[-1]

        _check_file_exists(temp_path, ('oof_prediction.npy', 'test_prediction.npy', 'metrics.txt'))


def test_experiment_lgb_multiclass():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  n_classes=5, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'multiclass',
        'max_depth': 8
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, X_test, temp_path)

        assert result.oof_prediction.shape == (len(y_train), 5)
        assert result.test_prediction.shape == (len(y_test), 5)

        _check_file_exists(temp_path, ('oof_prediction.npy', 'test_prediction.npy', 'metrics.txt'))


def test_experiment_cat_classifier():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id', target_name='tgt')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, X_test, temp_path, eval_func=roc_auc_score, gbdt_type='cat',
                                 submission_filename='submission.csv')

        assert roc_auc_score(y_train, result.oof_prediction) >= 0.85
        assert roc_auc_score(y_test, result.test_prediction) >= 0.85
        assert list(pd.read_csv(os.path.join(temp_path, 'submission.csv')).columns) == ['id', 'tgt']

        _check_file_exists(temp_path, ('submission.csv', 'oof_prediction.npy', 'test_prediction.npy', 'metrics.txt'))


def test_experiment_cat_regressor():
    X, y = make_regression_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                              random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, X_test, temp_path, gbdt_type='cat')

        assert mean_squared_error(y_train, result.oof_prediction) == result.metrics[-1]
        _check_file_exists(temp_path, ('oof_prediction.npy', 'test_prediction.npy', 'metrics.txt'))


def test_experiment_cat_multiclass():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2, n_classes=5,
                                  class_sep=0.98, random_state=0, id_column='user_id', target_name='tgt')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, X_test, temp_path, gbdt_type='cat',
                                 type_of_target='multiclass', submission_filename='submission.csv')

        assert result.oof_prediction.shape == (len(y_train), 5)
        assert result.test_prediction.shape == (len(y_test), 5)

        assert list(pd.read_csv(os.path.join(temp_path, 'submission.csv')).columns) == ['id', '0', '1', '2', '3', '4']

        _check_file_exists(temp_path, ('submission.csv', 'oof_prediction.npy', 'test_prediction.npy', 'metrics.txt'))


def test_experiment_cat_custom_eval():
    X, y = make_regression_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                              random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100,
        'eval_metric': 'MAE'
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, X_test, temp_path,
                                 gbdt_type='cat', eval_func=mean_absolute_error)

        assert mean_absolute_error(y_train, result.oof_prediction) == result.metrics[-1]
        _check_file_exists(temp_path, ('oof_prediction.npy', 'test_prediction.npy', 'metrics.txt'))


def test_experiment_without_test_data():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, None, temp_path)

        assert roc_auc_score(y_train, result.oof_prediction) >= 0.85
        _check_file_exists(temp_path, ('oof_prediction.npy', 'metrics.txt'))


def test_experiment_fit_params():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8,
        'n_estimators': 500
    }

    with get_temp_directory() as temp_path:
        result1 = experiment_gbdt(params, X_train, y_train, X_test,
                                  temp_path, fit_params={'early_stopping_rounds': None})
    with get_temp_directory() as temp_path:
        result2 = experiment_gbdt(params, X_train, y_train, X_test,
                                  temp_path, fit_params={'early_stopping_rounds': 5})

    assert result1.models[-1].booster_.num_trees() == params['n_estimators']
    assert result2.models[-1].booster_.num_trees() < params['n_estimators']


def test_experiment_mlflow():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with get_temp_directory() as temp_path:
        experiment_gbdt(params, X_train, y_train, None, temp_path, with_mlflow=True)

        _check_file_exists(temp_path, ('oof_prediction.npy', 'metrics.txt', 'mlflow.json'))

        # test if output files are also stored in the mlflow artifact uri
        with open(os.path.join(temp_path, 'mlflow.json'), 'r') as f:
            mlflow_meta = json.load(f)
            p = unquote(urlparse(mlflow_meta['artifact_uri']).path)
            if os.name == 'nt' and p.startswith("/"):
                p = p[1:]
            _check_file_exists(p, ('oof_prediction.npy', 'metrics.txt'))


def test_experiment_already_exists():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with get_temp_directory() as temp_path:
        experiment_gbdt(params, X_train, y_train, None, temp_path, overwrite=True)

        # result is overwrited by default
        experiment_gbdt(params, X_train, y_train, None, temp_path, overwrite=True)

        with pytest.raises(Exception):
            experiment_gbdt(params, X_train, y_train, None, temp_path, overwrite=False)


def test_submission_filename():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with get_temp_directory() as temp_path:
        experiment_gbdt(params, X_train, y_train, X_test, temp_path, submission_filename='sub.csv')

        df = pd.read_csv(os.path.join(temp_path, 'sub.csv'))
        assert list(df.columns) == ['id', 'target']


def test_experiment_manual_cv_kfold():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, None, temp_path, cv=KFold(4))
        assert len(result.models) == 4
        assert len(result.metrics) == 4 + 1


def test_experiment_manual_cv_int():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, None, temp_path, cv=KFold(2))
        assert len(result.models) == 2
        assert len(result.metrics) == 2 + 1


def test_experiment_manual_cv_group():
    df1 = pd.DataFrame()
    df1['x'] = np.random.randint(0, 10, size=1000)
    df1['y'] = df1['x'] > 5
    df1['grp'] = 0

    df2 = pd.DataFrame()
    df2['x'] = np.random.randint(0, 10, size=100)
    df2['y'] = df2['x'] <= 5
    df2['grp'] = 1

    X = pd.concat([df1, df2]).reset_index(drop=True)
    y = X['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    grp = X_train['grp']
    X_train = X_train.drop(['y', 'grp'], axis=1)
    X_test = X_test.drop(['y', 'grp'], axis=1)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, X_test, temp_path, cv=GroupKFold(2), groups=grp)
        assert result.metrics[-1] < 0.7


def test_find_best_parameter():
    params = {
        'objective': 'binary',
        'metrics': 'auc',
        'n_estimators': 1000
    }
    X, y = make_classification_df(2048, class_sep=0.7)
    X_train, X_test, y_train, y_test = train_test_split(X, y)


    best_params = find_best_lgbm_parameter(params, X_train, y_train, cv=5)

    result_base = experiment_gbdt(params, X_train, y_train, eval_func=roc_auc_score)
    result_opt = experiment_gbdt(best_params, X_train, y_train)

    assert result_opt.metrics[-1] >= result_base.metrics[-1]

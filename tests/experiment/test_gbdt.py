import json
import os
import pytest
import tempfile
from urllib.parse import urlparse, unquote

import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from nyaggle.experiment import experiment_gbdt
from nyaggle.testing import make_classification_df, make_regression_df


def _check_file_exists(directory, files):
    for f in files:
        assert os.path.exists(os.path.join(directory, f)), 'File not found: {}'.format(f)


def test_experiment_lgb_classifier():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with tempfile.TemporaryDirectory() as temp_path:
        result = experiment_gbdt(temp_path, params, 'user_id',
                                 X_train, y_train, X_test, roc_auc_score, stratified=True)

        assert roc_auc_score(y_train, result.predicted_oof) >= 0.85
        assert roc_auc_score(y_test, result.predicted_test) >= 0.85

        _check_file_exists(temp_path, ('submission.csv', 'oof.npy', 'test.npy', 'scores.txt'))


def test_experiment_lgb_regressor():
    X, y = make_regression_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                              random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    params = {
        'objective': 'regression',
        'max_depth': 8
    }

    with tempfile.TemporaryDirectory() as temp_path:
        result = experiment_gbdt(temp_path, params, 'user_id',
                                 X_train, y_train, X_test, stratified=True)

        assert mean_squared_error(y_train, result.predicted_oof) == result.scores[-1]

        _check_file_exists(temp_path, ('submission.csv', 'oof.npy', 'test.npy', 'scores.txt'))


def test_experiment_cat_classifier():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id', target_name='tgt')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    with tempfile.TemporaryDirectory() as temp_path:
        result = experiment_gbdt(temp_path, params, 'user_id',
                                 X_train, y_train, X_test, roc_auc_score, stratified=True, gbdt_type='cat')

        assert roc_auc_score(y_train, result.predicted_oof) >= 0.85
        assert roc_auc_score(y_test, result.predicted_test) >= 0.85
        assert list(pd.read_csv(os.path.join(temp_path, 'submission.csv')).columns) == ['user_id', 'tgt']

        _check_file_exists(temp_path, ('submission.csv', 'oof.npy', 'test.npy', 'scores.txt'))


def test_experiment_cat_regressor():
    X, y = make_regression_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                              random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    with tempfile.TemporaryDirectory() as temp_path:
        result = experiment_gbdt(temp_path, params, 'user_id',
                                 X_train, y_train, X_test, stratified=True, gbdt_type='cat')

        assert mean_squared_error(y_train, result.predicted_oof) == result.scores[-1]
        _check_file_exists(temp_path, ('submission.csv', 'oof.npy', 'test.npy', 'scores.txt'))


def test_experiment_cat_custom_eval():
    X, y = make_regression_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                              random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    params = {
        'max_depth': 8,
        'num_boost_round': 100,
        'eval_metric': 'MAE'
    }

    with tempfile.TemporaryDirectory() as temp_path:
        result = experiment_gbdt(temp_path, params, 'user_id',
                                 X_train, y_train, X_test, stratified=True, gbdt_type='cat', eval=mean_absolute_error)

        assert mean_absolute_error(y_train, result.predicted_oof) == result.scores[-1]
        _check_file_exists(temp_path, ('submission.csv', 'oof.npy', 'test.npy', 'scores.txt'))


def test_experiment_without_test_data():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with tempfile.TemporaryDirectory() as temp_path:
        result = experiment_gbdt(temp_path, params, 'user_id', X_train, y_train)

        assert roc_auc_score(y_train, result.predicted_oof) >= 0.85
        _check_file_exists(temp_path, ('oof.npy', 'scores.txt'))


def test_experiment_fit_params():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    params = {
        'objective': 'binary',
        'max_depth': 8,
        'n_estimators': 1000
    }

    with tempfile.TemporaryDirectory() as temp_path:
        result1 = experiment_gbdt(temp_path, params, 'user_id', X_train, y_train, X_test,
                                  fit_params={'early_stopping_rounds': None})
        result2 = experiment_gbdt(temp_path, params, 'user_id', X_train, y_train, X_test,
                                  fit_params={'early_stopping_rounds': 10})

        assert result1.scores[-1] != result2.scores[-1]


def test_experiment_seed_split():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with tempfile.TemporaryDirectory() as temp_path:
        result1 = experiment_gbdt(temp_path, params, 'user_id', X_train, y_train, X_test, seed_split=1)
        result2 = experiment_gbdt(temp_path, params, 'user_id', X_train, y_train, X_test, seed_split=1)
        result3 = experiment_gbdt(temp_path, params, 'user_id', X_train, y_train, X_test, seed_split=2)

        assert result1.scores[-1] == result2.scores[-1]
        assert result1.scores[-1] != result3.scores[-1]


def test_experiment_mlflow():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with tempfile.TemporaryDirectory() as temp_path:
        experiment_gbdt(temp_path, params, 'user_id', X_train, y_train, with_mlflow=True)

        _check_file_exists(temp_path, ('oof.npy', 'scores.txt', 'mlflow.json'))

        # test if output files are also stored in the mlflow artifact uri
        with open(os.path.join(temp_path, 'mlflow.json'), 'r') as f:
            mlflow_meta = json.load(f)
            p = unquote(urlparse(mlflow_meta['artifact_uri']).path)
            if os.name == 'nt' and p.startswith("/"):
                p = p[1:]
            _check_file_exists(p, ('oof.npy', 'scores.txt'))


def test_experiment_already_exists():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with tempfile.TemporaryDirectory() as temp_path:
        experiment_gbdt(temp_path, params, 'user_id', X_train, y_train)

        # result is overwrited by default
        experiment_gbdt(temp_path, params, 'user_id', X_train, y_train)

        with pytest.raises(Exception):
            experiment_gbdt(temp_path, params, 'user_id', X_train, y_train, overwrite=False)


def test_submission_filename():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with tempfile.TemporaryDirectory() as temp_path:
        experiment_gbdt(temp_path, params, 'user_id', X_train, y_train, X_test, submission_filename='sub.csv')

        df = pd.read_csv(os.path.join(temp_path, 'sub.csv'))
        assert list(df.columns) == ['user_id', 'target']


def test_stratified():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with tempfile.TemporaryDirectory() as temp_path:
        result1 = experiment_gbdt(temp_path, params, 'user_id', X_train, y_train, X_test, stratified=True)
        result2 = experiment_gbdt(temp_path, params, 'user_id', X_train, y_train, X_test, stratified=False)

        assert result1.scores[-1] != result2.scores[-1]

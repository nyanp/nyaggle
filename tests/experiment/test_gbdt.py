import json
import os
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, log_loss
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from nyaggle.experiment import experiment_gbdt, find_best_lgbm_parameter
from nyaggle.feature_store import save_feature, load_features
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

        assert len(np.unique(result.oof_prediction)) > 5  # making sure prediction is not binarized
        assert len(np.unique(result.test_prediction)) > 5
        assert roc_auc_score(y_train, result.oof_prediction) >= 0.9
        assert roc_auc_score(y_test, result.test_prediction) >= 0.9

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

        assert len(np.unique(result.oof_prediction)) > 5  # making sure prediction is not binarized
        assert len(np.unique(result.test_prediction)) > 5
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

        assert len(np.unique(result.oof_prediction[:, 0])) > 5  # making sure prediction is not binarized
        assert len(np.unique(result.test_prediction[:, 0])) > 5
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

        assert len(np.unique(result.oof_prediction)) > 5  # making sure prediction is not binarized
        assert len(np.unique(result.test_prediction)) > 5
        assert roc_auc_score(y_train, result.oof_prediction) >= 0.9
        assert roc_auc_score(y_test, result.test_prediction) >= 0.9
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


def test_experiment_xgb_classifier():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id', target_name='tgt')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, X_test, temp_path, eval_func=roc_auc_score, gbdt_type='xgb',
                                 submission_filename='submission.csv')

        assert len(np.unique(result.oof_prediction)) > 5  # making sure prediction is not binarized
        assert len(np.unique(result.test_prediction)) > 5
        assert roc_auc_score(y_train, result.oof_prediction) >= 0.9
        assert roc_auc_score(y_test, result.test_prediction) >= 0.9
        assert list(pd.read_csv(os.path.join(temp_path, 'submission.csv')).columns) == ['id', 'tgt']

        _check_file_exists(temp_path, ('submission.csv', 'oof_prediction.npy', 'test_prediction.npy', 'metrics.txt'))


def test_experiment_xgb_regressor():
    X, y = make_regression_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                              random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, X_test, temp_path, gbdt_type='xgb')

        assert mean_squared_error(y_train, result.oof_prediction) == result.metrics[-1]
        _check_file_exists(temp_path, ('oof_prediction.npy', 'test_prediction.npy', 'metrics.txt'))


def test_experiment_xgb_multiclass():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2, n_classes=5,
                                  class_sep=0.98, random_state=0, id_column='user_id', target_name='tgt')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, X_test, temp_path, gbdt_type='xgb',
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

        assert roc_auc_score(y_train, result.oof_prediction) >= 0.9
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


def test_experiment_sample_submission_binary():
    X, y = make_classification_df()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    sample_df = pd.DataFrame()
    sample_df['target_id_abc'] = np.arange(len(y_test)) + 10000
    sample_df['target_value_abc'] = 0

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, X_test, temp_path, sample_submission=sample_df)

        assert list(result.submission_df.columns) == ['target_id_abc', 'target_value_abc']
        assert roc_auc_score(y_test, result.submission_df['target_value_abc']) > 0.8


def test_experiment_sample_submission_multiclass():
    X, y = make_classification_df(n_classes=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    sample_df = pd.DataFrame()
    sample_df['target_id_abc'] = np.arange(len(y_test)) + 10000
    for i in range(5):
        sample_df['target_class_{}'.format(i)] = 0

    params = {
        'objective': 'multiclass',
        'max_depth': 8
    }

    with get_temp_directory() as temp_path:
        result = experiment_gbdt(params, X_train, y_train, X_test, temp_path, sample_submission=sample_df)

        assert list(result.submission_df.columns) == ['target_id_abc',
                                                      'target_class_0',
                                                      'target_class_1',
                                                      'target_class_2',
                                                      'target_class_3',
                                                      'target_class_4'
                                                      ]
        log_loss_trianed = log_loss(y_test, result.submission_df.drop('target_id_abc', axis=1), labels=[0, 1, 2, 3, 4])
        log_loss_default = log_loss(y_test, np.full((len(y_test), 5), 0.2), labels=[0, 1, 2, 3, 4])
        assert log_loss_trianed < log_loss_default


def test_with_feature_attachment():
    X, y = make_classification_df(n_num_features=5, class_sep=0.7)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with get_temp_directory() as temp_feature_path:
        cols = list(X.columns)
        for i, c in enumerate(cols):
            if X.shape[1] == 1:
                break
            save_feature(X[[c]], i, directory=temp_feature_path)
            X.drop(c, axis=1, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

        with get_temp_directory() as temp_path:
            result_wo_feature = experiment_gbdt(params, X_train, y_train, X_test, logging_directory=temp_path)

        with get_temp_directory() as temp_path:
            result_w_feature = experiment_gbdt(params, X_train, y_train, X_test, logging_directory=temp_path,
                                               feature_list=[0, 1, 2, 3], feature_directory=temp_feature_path)

        assert result_w_feature.metrics[-1] > result_wo_feature.metrics[-1]


def test_with_long_params():
    X, y = make_classification_df(1024, n_num_features=5, n_cat_features=400)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    with get_temp_directory() as temp_path:
        # just to make sure experiment finish
        experiment_gbdt(params, X_train, y_train, X_test,
                        logging_directory=temp_path, with_mlflow=True)

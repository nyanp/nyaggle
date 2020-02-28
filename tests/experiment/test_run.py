import json
import os
from urllib.parse import urlparse, unquote

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, log_loss
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from nyaggle.experiment import Experiment, run_experiment
from nyaggle.feature_store import save_feature
from nyaggle.testing import make_classification_df, make_regression_df, get_temp_directory


def _check_file_exists(directory, submission_filename=None, with_mlflow=False):
    files = ['oof_prediction.npy', 'test_prediction.npy', 'metrics.json', 'params.json']
    if submission_filename:
        files.append(submission_filename)
    if with_mlflow:
        files.append('mlflow.json')

    for f in files:
        assert os.path.exists(os.path.join(directory, f)), 'File not found: {}'.format(f)


def test_experiment_lgb_classifier(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name, eval_func=roc_auc_score)

    assert len(np.unique(result.oof_prediction)) > 5  # making sure prediction is not binarized
    assert len(np.unique(result.test_prediction)) > 5
    assert roc_auc_score(y_train, result.oof_prediction) >= 0.9
    assert roc_auc_score(y_test, result.test_prediction) >= 0.9

    _check_file_exists(tmpdir_name)


def test_experiment_lgb_regressor(tmpdir_name):
    X, y = make_regression_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                              random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'regression',
        'max_depth': 8
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name)

    assert len(np.unique(result.oof_prediction)) > 5  # making sure prediction is not binarized
    assert len(np.unique(result.test_prediction)) > 5
    assert mean_squared_error(y_train, result.oof_prediction) == result.metrics[-1]

    _check_file_exists(tmpdir_name)


def test_experiment_lgb_multiclass(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  n_classes=5, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'multiclass',
        'max_depth': 8
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name)

    assert len(np.unique(result.oof_prediction[:, 0])) > 5  # making sure prediction is not binarized
    assert len(np.unique(result.test_prediction[:, 0])) > 5
    assert result.oof_prediction.shape == (len(y_train), 5)
    assert result.test_prediction.shape == (len(y_test), 5)

    _check_file_exists(tmpdir_name)


def test_experiment_cat_classifier(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id', target_name='tgt')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name, eval_func=roc_auc_score, algorithm_type='cat',
                            submission_filename='submission.csv', with_auto_prep=True)

    assert len(np.unique(result.oof_prediction)) > 5  # making sure prediction is not binarized
    assert len(np.unique(result.test_prediction)) > 5
    assert roc_auc_score(y_train, result.oof_prediction) >= 0.9
    assert roc_auc_score(y_test, result.test_prediction) >= 0.9
    assert list(pd.read_csv(os.path.join(tmpdir_name, 'submission.csv')).columns) == ['id', 'tgt']

    _check_file_exists(tmpdir_name)


def test_experiment_cat_regressor(tmpdir_name):
    X, y = make_regression_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                              random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name, algorithm_type='cat')

    assert mean_squared_error(y_train, result.oof_prediction) == result.metrics[-1]
    _check_file_exists(tmpdir_name)


def test_experiment_cat_multiclass(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2, n_classes=5,
                                  class_sep=0.98, random_state=0, id_column='user_id', target_name='tgt')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name, algorithm_type='cat',
                            type_of_target='multiclass', submission_filename='submission.csv', with_auto_prep=True)

    assert result.oof_prediction.shape == (len(y_train), 5)
    assert result.test_prediction.shape == (len(y_test), 5)

    assert list(pd.read_csv(os.path.join(tmpdir_name, 'submission.csv')).columns) == ['id', '0', '1', '2', '3', '4']

    _check_file_exists(tmpdir_name, submission_filename='submission.csv')


def test_experiment_xgb_classifier(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id', target_name='tgt')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name, eval_func=roc_auc_score, algorithm_type='xgb',
                            submission_filename='submission.csv', with_auto_prep=True)

    assert len(np.unique(result.oof_prediction)) > 5  # making sure prediction is not binarized
    assert len(np.unique(result.test_prediction)) > 5
    assert roc_auc_score(y_train, result.oof_prediction) >= 0.9
    assert roc_auc_score(y_test, result.test_prediction) >= 0.9
    assert list(pd.read_csv(os.path.join(tmpdir_name, 'submission.csv')).columns) == ['id', 'tgt']

    _check_file_exists(tmpdir_name, submission_filename='submission.csv')


def test_experiment_xgb_regressor(tmpdir_name):
    X, y = make_regression_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                              random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name, algorithm_type='xgb', with_auto_prep=True)

    assert mean_squared_error(y_train, result.oof_prediction) == result.metrics[-1]
    _check_file_exists(tmpdir_name)


def test_experiment_xgb_multiclass(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2, n_classes=5,
                                  class_sep=0.98, random_state=0, id_column='user_id', target_name='tgt')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name, algorithm_type='xgb',
                            type_of_target='multiclass', submission_filename='submission.csv',
                            with_auto_prep=True)

    assert result.oof_prediction.shape == (len(y_train), 5)
    assert result.test_prediction.shape == (len(y_test), 5)

    assert list(pd.read_csv(os.path.join(tmpdir_name, 'submission.csv')).columns) == ['id', '0', '1', '2', '3', '4']

    _check_file_exists(tmpdir_name, submission_filename='submission.csv')


def test_experiment_sklearn_classifier(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=0,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'C': 0.1
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name, eval_func=roc_auc_score,
                            algorithm_type=LogisticRegression, with_auto_prep=False)

    assert len(np.unique(result.oof_prediction)) > 5  # making sure prediction is not binarized
    assert len(np.unique(result.test_prediction)) > 5
    assert roc_auc_score(y_train, result.oof_prediction) >= 0.8
    assert roc_auc_score(y_test, result.test_prediction) >= 0.8

    _check_file_exists(tmpdir_name)


def test_experiment_sklearn_regressor(tmpdir_name):
    X, y = make_regression_df(n_samples=1024, n_num_features=10, n_cat_features=0,
                              random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'fit_intercept': True
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name, with_auto_prep=False,
                            algorithm_type=LinearRegression)

    assert len(np.unique(result.oof_prediction)) > 5  # making sure prediction is not binarized
    assert len(np.unique(result.test_prediction)) > 5
    assert mean_squared_error(y_train, result.oof_prediction) == result.metrics[-1]

    _check_file_exists(tmpdir_name)


def test_experiment_sklearn_multiclass(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=0,
                                  n_classes=5, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'n_neighbors': 10
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name, algorithm_type=KNeighborsClassifier,
                            with_auto_prep=False)

    assert len(np.unique(result.oof_prediction[:, 0])) > 5  # making sure prediction is not binarized
    assert len(np.unique(result.test_prediction[:, 0])) > 5
    assert result.oof_prediction.shape == (len(y_train), 5)
    assert result.test_prediction.shape == (len(y_test), 5)

    _check_file_exists(tmpdir_name)


def test_experiment_cat_custom_eval(tmpdir_name):
    X, y = make_regression_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                              random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'max_depth': 8,
        'num_boost_round': 100,
        'eval_metric': 'MAE'
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name,
                            algorithm_type='cat', eval_func=mean_absolute_error)

    assert mean_absolute_error(y_train, result.oof_prediction) == result.metrics[-1]
    _check_file_exists(tmpdir_name)


def test_experiment_without_test_data(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    result = run_experiment(params, X_train, y_train, None, tmpdir_name)

    assert roc_auc_score(y_train, result.oof_prediction) >= 0.9
    _check_file_exists(tmpdir_name)


def test_experiment_fit_params(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8,
        'n_estimators': 500
    }

    result1 = run_experiment(params, X_train, y_train, X_test,
                             os.path.join(tmpdir_name, '1'), fit_params={'early_stopping_rounds': None})
    result2 = run_experiment(params, X_train, y_train, X_test,
                             os.path.join(tmpdir_name, '2'), fit_params={'early_stopping_rounds': 5})

    assert result1.models[-1].booster_.num_trees() == params['n_estimators']
    assert result2.models[-1].booster_.num_trees() < params['n_estimators']


def test_experiment_mlflow(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    run_experiment(params, X_train, y_train, None, tmpdir_name, with_mlflow=True)

    _check_file_exists(tmpdir_name, with_mlflow=True)

    # test if output files are also stored in the mlflow artifact uri
    with open(os.path.join(tmpdir_name, 'mlflow.json'), 'r') as f:
        mlflow_meta = json.load(f)
        p = unquote(urlparse(mlflow_meta['artifact_uri']).path)
        if os.name == 'nt' and p.startswith("/"):
            p = p[1:]
        _check_file_exists(p, with_mlflow=False)


def test_experiment_already_exists(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    run_experiment(params, X_train, y_train, None, tmpdir_name)

    # result is not overwrited by default
    run_experiment(params, X_train, y_train, None, tmpdir_name, if_exists='replace')

    with pytest.raises(Exception):
        run_experiment(params, X_train, y_train, None, tmpdir_name)


def test_submission_filename(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    run_experiment(params, X_train, y_train, X_test, tmpdir_name, submission_filename='sub.csv')

    df = pd.read_csv(os.path.join(tmpdir_name, 'sub.csv'))
    assert list(df.columns) == ['id', 'target']


def test_experiment_manual_cv_kfold(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    result = run_experiment(params, X_train, y_train, None, tmpdir_name, cv=KFold(4))
    assert len(result.models) == 4
    assert len(result.metrics) == 4 + 1


def test_experiment_manual_cv_int(tmpdir_name):
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    result = run_experiment(params, X_train, y_train, None, tmpdir_name, cv=KFold(2))
    assert len(result.models) == 2
    assert len(result.metrics) == 2 + 1


def test_experiment_manual_cv_group(tmpdir_name):
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

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name, cv=GroupKFold(2), groups=grp)
    assert result.metrics[-1] < 0.7


def test_experiment_sample_submission_binary(tmpdir_name):
    X, y = make_classification_df()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    sample_df = pd.DataFrame()
    sample_df['target_id_abc'] = np.arange(len(y_test)) + 10000
    sample_df['target_value_abc'] = 0

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name, sample_submission=sample_df)

    assert list(result.submission_df.columns) == ['target_id_abc', 'target_value_abc']
    assert roc_auc_score(y_test, result.submission_df['target_value_abc']) > 0.8


def test_experiment_sample_submission_multiclass(tmpdir_name):
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

    result = run_experiment(params, X_train, y_train, X_test, tmpdir_name, sample_submission=sample_df)

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


def test_with_feature_attachment(tmpdir_name):
    X, y = make_classification_df(n_num_features=5, class_sep=0.7)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    cols = list(X.columns)
    for i, c in enumerate(cols):
        if X.shape[1] == 1:
            break
        save_feature(X[[c]], i, directory=tmpdir_name)
        X.drop(c, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    result_wo_feature = run_experiment(params, X_train, y_train, X_test,
                                       logging_directory=os.path.join(tmpdir_name, 'wo'))

    result_w_feature = run_experiment(params, X_train, y_train, X_test,
                                      logging_directory=os.path.join(tmpdir_name, 'w'),
                                      feature_list=[0, 1, 2, 3], feature_directory=tmpdir_name)

    assert result_w_feature.metrics[-1] > result_wo_feature.metrics[-1]


def test_with_long_params(tmpdir_name):
    X, y = make_classification_df(1024, n_num_features=5, n_cat_features=400)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    # just to make sure experiment finish
    run_experiment(params, X_train, y_train, X_test,
                   logging_directory=tmpdir_name, with_mlflow=True)


@pytest.mark.parametrize('cat_cast', [True, False])
@pytest.mark.parametrize('algorithm', ['cat', 'xgb', 'lgbm'])
def test_with_rare_categories(tmpdir_name, cat_cast, algorithm):
    X = pd.DataFrame({
        'x0': [None]*100,
        'x1': np.random.choice([np.inf, -np.inf], size=100),
        'x2': ['nan'] + [None]*99,
        'x3': np.concatenate([np.random.choice(['A', 'B'], size=50), np.random.choice(['C', 'D', 'na'], size=50)])
    })

    y = pd.Series(np.random.choice([0, 1], size=100), name='y')

    params = {
        'lgbm': {
            'objective': 'binary',
            'max_depth': 8
        },
        'xgb': {
            'objective': 'binary:logistic',
            'max_depth': 8
        },
        'cat': {
            'loss_function': 'Logloss',
            'max_depth': 8
        }
    }

    X_ = X.copy()
    y_ = y.copy()
    if cat_cast:
        for c in X.columns:
            X_[c] = X_[c].astype('category')
        X_ = X_.iloc[:50, :]
        y_ = y_.iloc[:50]

    X_train, X_test, y_train, y_test = train_test_split(X_, y_, shuffle=False, test_size=0.5)

    run_experiment(params[algorithm], X_train, y_train, X_test, algorithm_type=algorithm,
                   cv=StratifiedKFold(3, shuffle=True, random_state=0),
                   logging_directory=tmpdir_name,
                   with_mlflow=True, with_auto_prep=True,
                   categorical_feature=['x0', 'x1', 'x2', 'x3'])


def test_inherit_outer_scope_run(tmpdir_name):
    mlflow.start_run()
    mlflow.log_param('foo', 1)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }
    X, y = make_classification_df()

    run_experiment(params, X, y, with_mlflow=True, logging_directory=tmpdir_name)

    assert mlflow.active_run() is not None  # still valid

    client = mlflow.tracking.MlflowClient()
    data = client.get_run(mlflow.active_run().info.run_id).data

    assert data.metrics['Overall'] > 0  # recorded

    mlflow.end_run()


def test_custom_experiment(tmpdir_name):
    params = {
        'objective': 'binary',
        'max_depth': 8
    }
    X, y = make_classification_df()

    with Experiment(tmpdir_name, with_mlflow=True) as e:
        run_experiment(params, X, y, logging_directory='foobar', inherit_experiment=e)

    # all files are logged into e.logging_directory, instead of 'foobar'
    _check_file_exists(tmpdir_name, with_mlflow=True)

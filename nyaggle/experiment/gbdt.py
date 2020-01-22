import copy
import os
import time
from collections import namedtuple
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import optuna.integration.lightgbm as optuna_lgb
import pandas as pd
import sklearn.utils.multiclass as multiclass
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMModel, LGBMClassifier, LGBMRegressor
from more_itertools import first_true
from pandas.api.types import is_integer_dtype, is_categorical
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss

from nyaggle.experiment.experiment import Experiment
from nyaggle.util import plot_importance
from nyaggle.validation.cross_validate import cross_validate
from nyaggle.validation.split import check_cv

GBDTResult = namedtuple('LGBResult', ['oof_prediction', 'test_prediction', 'metrics', 'models', 'importance', 'time',
                                      'submission_df'])


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
    print(tuning_history)

    result_param = copy.deepcopy(base_param)
    for p in best_params:
        result_param[p] = best_params[p]
    return result_param


def experiment_gbdt(model_params: Dict[str, Any],
                    X_train: pd.DataFrame, y: pd.Series,
                    X_test: Optional[pd.DataFrame] = None,
                    logging_directory: str = 'output/{time}',
                    overwrite: bool = False,
                    eval_func: Optional[Callable] = None,
                    gbdt_type: str = 'lgbm',
                    fit_params: Optional[Dict[str, Any]] = None,
                    cv: Optional[Union[int, Iterable, BaseCrossValidator]] = None,
                    groups: Optional[pd.Series] = None,
                    categorical_feature: Optional[List[str]] = None,
                    sample_submission: Optional[pd.DataFrame] = None,
                    submission_filename: Optional[str] = None,
                    type_of_target: str = 'auto',
                    tuning_time_budget: Optional[int] = None,
                    with_auto_prep: bool = True,
                    with_mlflow: bool = False,
                    mlflow_experiment_id: Optional[Union[int, str]] = None,
                    mlflow_run_name: Optional[str] = None,
                    mlflow_tracking_uri: Optional[str] = None
                    ):
    """
    Evaluate metrics by cross-validation and stores result
    (log, oof prediction, test prediction, feature importance plot and submission file)
    under the directory specified.

    One of the following estimators are used (automatically dispatched by ``type_of_target(y)`` and ``gbdt_type``).

    * LGBMClassifier
    * LGBMRegressor
    * CatBoostClassifier
    * CatBoostRegressor

    The output files are laid out as follows:

    .. code-block:: none

      <logging_directory>/
          log.txt                  <== Logging file
          importance.png           <== Feature importance plot generated by nyaggle.util.plot_importance
          oof_prediction.npy       <== Out of fold prediction in numpy array format
          test_prediction.npy      <== Test prediction in numpy array format
          submission.csv           <== Submission csv file
          metrics.txt              <== Metrics
          params.txt               <== Parameters
          models/
              fold1                <== The trained model in fold 1
              ...

    Args:
        model_params:
            Parameters passed to the constructor of the classifier/regressor object (i.e. LGBMRegressor).
        X_train:
            Training data. Categorical feature should be casted to pandas categorical type or encoded to integer.
        y:
            Target
        X_test:
            Test data (Optional). If specified, prediction on the test data is performed using ensemble of models.
        logging_directory:
            Path to directory where output of experiment is stored.
        overwrite:
            If True, contents in ``logging_directory`` will be overwritten.
        fit_params:
            Parameters passed to the fit method of the estimator.
        eval_func:
            Function used for logging and calculation of returning scores.
            This parameter isn't passed to GBDT, so you should set objective and eval_metric separately if needed.
            If ``eval_func`` is None, ``roc_auc_score`` or ``mean_squared_error`` is used by default.
        gbdt_type:
            Type of gradient boosting library used. "lgbm" (lightgbm) or "cat" (catboost)
        cv:
            int, cross-validation generator or an iterable which determines the cross-validation splitting strategy.

            - None, to use the default ``KFold(5, random_state=0, shuffle=True)``,
            - integer, to specify the number of folds in a ``(Stratified)KFold``,
            - CV splitter (the instance of ``BaseCrossValidator``),
            - An iterable yielding (train, test) splits as arrays of indices.
        groups:
            Group labels for the samples. Only used in conjunction with a “Group” cv instance (e.g., ``GroupKFold``).
        sample_submission:
            A sample dataframe alined with test data (Usually in Kaggle, it is available as sample_submission.csv).
            The submission file will be created with the same schema as this dataframe.
        submission_filename:
            The name of submission file created under logging directory. If ``None``, the basename of the logging
            directory will be used as a filename.
        categorical_feature:
            List of categorical column names. If ``None``, categorical columns are automatically determined by dtype.
        type_of_target:
            The type of target variable. If ``auto``, type is inferred by ``sklearn.utils.multiclass.type_of_target``.
            Otherwise, ``binary``, ``continuous``, or ``multiclass`` are supported.
        tuning_time_budget:
            If not ``None``, model parameters will be automatically updated using optuna with the specified time
             budgets in seconds (only available in lightgbm).
        with_auto_prep:
            If True, the input datasets will be copied and automatic preprocessing will be performed on them.
            For example, if ``gbdt_type = 'cat'``, all missing values in categorical features will be filled.
        with_mlflow:
            If True, `mlflow tracking <https://www.mlflow.org/docs/latest/tracking.html>`_ is used.
            One instance of ``nyaggle.experiment.Experiment`` corresponds to one run in mlflow.
            Note that all output
            mlflow's directory (``mlruns`` by default).
        mlflow_experiment_id:
            ID of the experiment of mlflow. Passed to ``mlflow.start_run()``.
        mlflow_run_name:
            Name of the run in mlflow. Passed to ``mlflow.start_run()``.
            If ``None``, ``logging_directory`` is used as the run name.
        mlflow_tracking_uri:
            Tracking server uri in mlflow. Passed to ``mlflow.set_tracking_uri``.
    :return:
        Namedtuple with following members

        * oof_prediction:
            numpy array, shape (len(X_train),) Predicted value on Out-of-Fold validation data.
        * test_prediction:
            numpy array, shape (len(X_test),) Predicted value on test data. ``None`` if X_test is ``None``
        * metrics:
            list of float, shape(nfolds+1) ``scores[i]`` denotes validation score in i-th fold.
            ``scores[-1]`` is overall score.
        * models:
            list of objects, shape(nfolds) Trained models for each folds.
        * importance:
            list of pd.DataFrame, feature importance for each fold (type="gain").
        * time:
            Training time in seconds.
        * submit_df:
            The dataframe saved as submission.csv
    """
    start_time = time.time()
    cv = check_cv(cv, y)

    _check_input(X_train, y, X_test)
    if with_auto_prep:
        X_train, X_test = autoprep(X_train, X_test, categorical_feature, gbdt_type)

    logging_directory = logging_directory.format(time=datetime.now().strftime('%Y%m%d_%H%M%S'))

    with Experiment(logging_directory, overwrite, with_mlflow=with_mlflow, mlflow_tracking_uri=mlflow_tracking_uri,
                    mlflow_experiment_id=mlflow_experiment_id, mlflow_run_name=mlflow_run_name) as exp:
        exp.log('GBDT: {}'.format(gbdt_type))
        exp.log('Experiment: {}'.format(logging_directory))
        exp.log('Params: {}'.format(model_params))
        exp.log('Features: {}'.format(list(X_train.columns)))
        exp.log_param('gbdt_type', gbdt_type)
        exp.log_param('num_features', X_train.shape[1])
        exp.log_param('fit_params', fit_params)
        exp.log_param('model_params', model_params)

        if tuning_time_budget is not None:
            assert gbdt_type == 'lgbm', 'auto-tuning with catboost is not supported'
            model_params = find_best_lgbm_parameter(model_params, X_train, y, cv=cv, groups=groups,
                                                    time_budget=tuning_time_budget, type_of_target=type_of_target)
            exp.log_param('model_params_tuned', model_params)

        if categorical_feature is None:
            categorical_feature = [c for c in X_train.columns if X_train[c].dtype.name in ['object', 'category']]
        exp.log('Categorical: {}'.format(categorical_feature))

        if type_of_target == 'auto':
            type_of_target = multiclass.type_of_target(y)
        model, eval_func, cat_param_name = _dispatch_gbdt(gbdt_type, type_of_target, eval_func)
        models = [model(**model_params) for _ in range(cv.get_n_splits())]

        if fit_params is None:
            fit_params = {}
        if cat_param_name is not None and cat_param_name not in fit_params:
            fit_params[cat_param_name] = categorical_feature

        exp.log_params(fit_params)

        predict_proba = type_of_target == 'multiclass'
        result = cross_validate(models, X_train=X_train, y=y, X_test=X_test, cv=cv, groups=groups,
                                logger=exp.get_logger(), eval_func=eval_func, fit_params=fit_params,
                                predict_proba=predict_proba, type_of_target=type_of_target)

        # save oof
        exp.log_numpy('oof_prediction', result.oof_prediction)
        exp.log_numpy('test_prediction', result.test_prediction)

        for i in range(cv.get_n_splits()):
            exp.log_metric('Fold {}'.format(i + 1), result.scores[i])
        exp.log_metric('Overall', result.scores[-1])

        # save importance plot
        importance = pd.concat(result.importance)
        plot_file_path = os.path.join(logging_directory, 'importance.png')
        plot_importance(importance, plot_file_path)
        exp.log_artifact(plot_file_path)

        # save trained model
        for i, model in enumerate(models):
            _save_model(gbdt_type, model, logging_directory, i + 1, exp)

        # save submission.csv
        submit_df = None
        if X_test is not None:
            if sample_submission is not None:
                submit_df = sample_submission.copy()

                if type_of_target == 'multiclass':
                    n_id_cols = submit_df.shape[1] - result.test_prediction.shape[1]
                    for i, y in enumerate(sorted(y.unique())):
                        submit_df.iloc[:, n_id_cols + i] = result.test_prediction[:, i]
                else:
                    submit_df.iloc[:, -1] = result.test_prediction
            else:
                submit_df = pd.DataFrame()
                submit_df['id'] = np.arange(len(X_test))

                if type_of_target == 'multiclass':
                    for i, y in enumerate(sorted(y.unique())):
                        submit_df[y] = result.test_prediction[:, i]
                else:
                    submit_df[y.name] = result.test_prediction

            if submission_filename is None:
                submission_filename = os.path.basename(logging_directory)

            exp.log_dataframe(submission_filename, submit_df, 'csv')

        elapsed_time = time.time() - start_time

        return GBDTResult(result.oof_prediction, result.test_prediction,
                          result.scores, models, result.importance, elapsed_time, submit_df)


def _dispatch_gbdt(gbdt_type: str, target_type: str, custom_eval: Optional[Callable] = None):
    gbdt_table = [
        ('binary', 'lgbm', LGBMClassifier, roc_auc_score, 'categorical_feature'),
        ('multiclass', 'lgbm', LGBMClassifier, log_loss, 'categorical_feature'),
        ('continuous', 'lgbm', LGBMRegressor, mean_squared_error, 'categorical_feature'),
        ('binary', 'cat', CatBoostClassifier, roc_auc_score, 'cat_features'),
        ('multiclass', 'cat', CatBoostClassifier, log_loss, 'cat_features'),
        ('continuous', 'cat', CatBoostRegressor, mean_squared_error, 'cat_features'),
    ]
    found = first_true(gbdt_table, pred=lambda x: x[0] == target_type and x[1] == gbdt_type)
    if found is None:
        raise RuntimeError('Not supported gbdt_type ({}) or type_of_target ({}).'.format(gbdt_type, target_type))

    model, eval_func, cat_param = found[2], found[3], found[4]
    if custom_eval is not None:
        eval_func = custom_eval

    return model, eval_func, cat_param


def _save_model(gbdt_type: str, model: Union[CatBoost, LGBMModel], logging_directory: str, fold: int, exp: Experiment):
    model_dir = os.path.join(logging_directory, 'models')
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, 'fold{}'.format(fold))

    if gbdt_type == 'cat':
        assert isinstance(model, CatBoost)
        model.save_model(path)
    else:
        assert isinstance(model, LGBMModel)
        model.booster_.save_model(path)

    exp.log_artifact(path)


def _check_input(X_train: pd.DataFrame, y: pd.Series,
                 X_test: Optional[pd.DataFrame] = None):
    assert len(X_train) == len(y), "length of X_train and y are different. len(X_train) = {}, len(y) = {}".format(
        len(X_train), len(y)
    )

    if X_test is not None:
        assert list(X_train.columns) == list(X_test.columns), "columns are different between X_train and X_test"


def autoprep(X_train: pd.DataFrame, X_test: Optional[pd.DataFrame],
             categorical_feature: Optional[List[str]] = None,
             gbdt_type: str = 'lgbm') -> Tuple[pd.DataFrame, pd.DataFrame]:
    if categorical_feature is None:
        categorical_feature = [c for c in X_train.columns if X_train[c].dtype.name in ['object', 'category']]

    if gbdt_type == 'cat' and len(categorical_feature) > 0:
        X_train = X_train.copy()
        X_all = X_train if X_test is None else pd.concat([X_train, X_test])

        # https://catboost.ai/docs/concepts/faq.html#why-float-and-nan-values-are-forbidden-for-cat-features
        for c in categorical_feature:
            if is_integer_dtype(X_train[c].dtype):
                fillval = X_all[c].min() - 1
            else:
                unique_values = X_all[c].unique()
                fillval = 'na'
                while fillval in unique_values:
                    fillval += '-'
            if is_categorical(X_train[c]):
                X_train[c] = X_train[c].cat.add_categories(fillval).fillna(fillval)
                if X_test is not None:
                    X_test[c] = X_test[c].cat.add_categories(fillval).fillna(fillval)
            else:
                X_train[c].fillna(fillval, inplace=True)
                if X_test is not None:
                    X_test[c].fillna(fillval, inplace=True)

    return X_train, X_test
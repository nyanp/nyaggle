import os
from collections import namedtuple
from logging import getLogger, FileHandler, DEBUG
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.utils.multiclass import type_of_target

from nyaggle.model.cv import cv
from nyaggle.util import plot_importance


LGBResult = namedtuple('LGBResult', ['predicted_oof', 'predicted_test', 'scores', 'importance'])


def experiment_lgb(logging_directory: str, lgb_params: Dict[str, Any], id_column: str,
                   X_train: pd.DataFrame, y: pd.Series,
                   X_test: Optional[pd.DataFrame] = None,
                   eval: Optional[Callable] = None, 
                   nfolds: int = 5,
                   overwrite: bool = True,
                   stratified: bool = False,
                   seed: int = 42,
                   submission_filename: str = 'submission.csv'):
    """
    Evaluate metrics by cross-validation and stores result
    (log, oof prediction, test prediction, feature importance plot and submission file)
    under the directory specified.

    LGBMClassifier or LGBMRegressor with early-stopping is used (dispatched by ``type_of_target(y)``).

    Args:
        logging_directory:
            Path to directory where output of experiment is stored.
        lgb_params:
            Parameters passed to the constructor of LGBMClassifer/LGBMRegressor.
        id_column:
            The name of index or column which is used as index.
            If `X_test` is not None, submission file is created along with this column.
        X_train:
            Training data
        y:
            Target
        X_test:
            Test data (Optional). If specified, prediction on the test data is performed using ensemble of models.
        eval:
            Function used for logging and returning scores
        nfolds:
            Number of splits
        overwrite:
            If True, contents in ``logging_directory`` will be overwritten.
        stratified:
            If true, use stratified K-Fold
        seed:
            Seed used by the random number generator in ``KFold``
        submission_filename:
            The name of submission file created under logging directory.
    :return:
        Namedtuple with following members

        * predicted_oof:
            numpy array, shape (len(X_train),) Predicted value on Out-of-Fold validation data.
        * predicted_test:
            numpy array, shape (len(X_test),) Predicted value on test data. ``None`` if X_test is ``None``
        * scores:
            list of float, shape(nfolds+1) ``scores[i]`` denotes validation score in i-th fold.
            ``scores[-1]`` is overall score. `None` if eval is not specified
        * importance:
            pd.DataFrame, feature importance (average over folds, type="gain").
    """
    if id_column in X_train.columns:
        if X_test is not None:
            assert list(X_train.columns) == list(X_test.columns)
            X_test.set_index(id_column, inplace=True)
        X_train.set_index(id_column, inplace=True)
        
    assert X_train.index.name == id_column, "index does not match"

    os.makedirs(logging_directory, exist_ok=overwrite)

    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    logger.addHandler(FileHandler(os.path.join(logging_directory, 'log.txt')))

    logger.info('Experiment: {}'.format(logging_directory))
    logger.info('Params: {}'.format(lgb_params))
    logger.info('Features: {}'.format(list(X_train.columns)))

    categorical_feature = [c for c in X_train.columns if X_train[c].dtype.name in ['object', 'category']]
    logger.info('Categorical: {}'.format(categorical_feature))

    if type_of_target(y) == 'binary':
        models = [LGBMClassifier(**lgb_params)] * nfolds
        if eval is None:
            eval = roc_auc_score
    else:
        models = [LGBMRegressor(**lgb_params)] * nfolds
        if eval is None:
            eval = mean_squared_error

    importances = []

    def callback(fold: int, model: LGBMClassifier, x_train: pd.DataFrame, y: pd.Series):
        df = pd.DataFrame({
            'feature': list(x_train.columns),
            'importance': model.booster_.feature_importance(importance_type='gain')
        })
        importances.append(df)

    result = cv(models, X_train=X_train, y=y, X_test=X_test, nfolds=nfolds, logger=logger,
                on_each_fold=callback, eval=eval, stratified=stratified, seed=seed,
                fit_params={'categorical_feature': categorical_feature})

    importance = pd.concat(importances)

    importance = importance.groupby('feature')['importance'].mean().reset_index()
    importance.sort_values(by='importance', ascending=False, inplace=True)

    plot_importance(importance, os.path.join(logging_directory, 'feature_importance.png'))

    # save oof
    np.save(os.path.join(logging_directory, 'oof'), result.predicted_oof)
    np.save(os.path.join(logging_directory, 'test'), result.predicted_test)

    submit = pd.DataFrame({
        id_column: X_test.index,
        y.name: result.predicted_test
    })
    submit.to_csv(os.path.join(logging_directory, submission_filename), index=False)

    return LGBResult(result.predicted_oof, result.predicted_test, result.scores, importance)

import copy
import os
import pytest
import tempfile

import lightgbm as lgb
import pandas as pd
from pandas.testing import assert_frame_equal

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import category_encoders as ce

from nyaggle.experiment import experiment_lgb
from nyaggle.testing import make_classification_df


def test_experiment_lgb_classifier():
    X, y = make_classification_df(n_samples=1024, n_features=20, class_sep=0.98, random_state=0, id_column='user_id')

    X_train = X.iloc[:512, :]
    y_train = y.iloc[:512]
    X_test = X.iloc[512:, :]
    y_test = y.iloc[512:]

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with tempfile.TemporaryDirectory() as temp_path:
        result = experiment_lgb(temp_path, params, 'user_id',
                                X_train, y_train, X_test, roc_auc_score, stratified=True)

        assert roc_auc_score(y_train, result.predicted_oof) >= 0.85
        assert roc_auc_score(y_test, result.predicted_test) >= 0.85

        assert os.path.exists(os.path.join(temp_path, 'submission.csv'))

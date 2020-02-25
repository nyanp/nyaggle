import numpy as np
import scipy.stats as stats
from numpy.testing import assert_array_almost_equal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.utils.multiclass import type_of_target
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score, mean_squared_error

from nyaggle.testing import make_classification_df, make_regression_df
from nyaggle.ensemble import stacking
from nyaggle.validation import cross_validate


def _make_1st_stage_preds(X, y, X_test):
    if type_of_target(y) == 'continuous':
        models = [
            SVR(),
            Ridge(random_state=0),
            RandomForestRegressor(n_estimators=30, random_state=0)
        ]
    else:
        models = [
            SVC(random_state=0),
            LogisticRegression(random_state=0),
            RandomForestClassifier(n_estimators=30, random_state=0)
        ]

    results = [cross_validate(m, X, y, X_test, cv=5) for m in models]

    return [r.oof_prediction for r in results], [r.test_prediction for r in results]


def test_stacking():
    X, y = make_classification_df()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    oof, test = _make_1st_stage_preds(X_train, y_train, X_test)

    worst_base_roc = min(roc_auc_score(y_train, oof[0]), roc_auc_score(y_train, oof[1]), roc_auc_score(y_train, oof[2]))

    result = stacking(test, oof, y_train, eval_func=roc_auc_score)

    assert roc_auc_score(y_train, result.oof_prediction) > worst_base_roc



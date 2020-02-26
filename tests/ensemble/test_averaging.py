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
from nyaggle.ensemble import averaging, averaging_opt
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


def test_averaging():
    X, y = make_classification_df()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    _, test = _make_1st_stage_preds(X_train, y_train, X_test)

    result = averaging(test)

    assert_array_almost_equal((test[0]+test[1]+test[2])/3, result.test_prediction)
    assert result.score is None
    assert result.oof_prediction is None


def test_averaging_with_oof():
    X, y = make_classification_df()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    oof, test = _make_1st_stage_preds(X_train, y_train, X_test)

    result = averaging(test, oof, y_train)

    assert_array_almost_equal((test[0]+test[1]+test[2])/3, result.test_prediction)
    assert_array_almost_equal((oof[0]+oof[1]+oof[2])/3, result.oof_prediction)
    assert result.score is None


def test_averaging_regression():
    X, y = make_regression_df()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    oof, test = _make_1st_stage_preds(X_train, y_train, X_test)

    result = averaging(test, oof, y_train)

    assert_array_almost_equal((test[0]+test[1]+test[2])/3, result.test_prediction)
    assert_array_almost_equal((oof[0]+oof[1]+oof[2])/3, result.oof_prediction)
    assert result.score is None


def test_averaging_multiclass():
    X, y = make_classification_df(n_classes=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    oof, test = _make_1st_stage_preds(X_train, y_train, X_test)

    result = averaging(test, oof, y_train)

    assert_array_almost_equal((test[0]+test[1]+test[2])/3, result.test_prediction)
    assert_array_almost_equal((oof[0]+oof[1]+oof[2])/3, result.oof_prediction)
    assert result.score is None


def test_averaging_with_metrics():
    X, y = make_classification_df()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    oof, test = _make_1st_stage_preds(X_train, y_train, X_test)

    result = averaging(test, oof, y_train, eval_func=roc_auc_score)

    assert result.score == roc_auc_score(y_train, result.oof_prediction)


def test_weight_averaging():
    X, y = make_classification_df()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    oof, test = _make_1st_stage_preds(X_train, y_train, X_test)

    result = averaging(test, oof, y_train, weights=[0.2, 0.4, 0.3])

    assert_array_almost_equal(0.2*test[0]+0.4*test[1]+0.3*test[2], result.test_prediction)
    assert_array_almost_equal(0.2*oof[0]+0.4*oof[1]+0.3*oof[2], result.oof_prediction)
    assert result.score is None


def test_rank_averaging():
    X, y = make_classification_df(n_samples=1024)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    oof, test = _make_1st_stage_preds(X_train, y_train, X_test)

    result = averaging(test, rank_averaging=True)

    test_rank = [stats.rankdata(t) / len(X_test) for t in test]

    assert_array_almost_equal((test_rank[0]+test_rank[1]+test_rank[2])/3, result.test_prediction)
    assert result.score is None


def test_rank_averaging_with_oof():
    X, y = make_classification_df(n_samples=1024)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    oof, test = _make_1st_stage_preds(X_train, y_train, X_test)

    result = averaging(test, oof, y_train, rank_averaging=True)

    oof_rank = [stats.rankdata(o) / len(X_train) for o in oof]
    test_rank = [stats.rankdata(t) / len(X_test) for t in test]

    assert_array_almost_equal((test_rank[0]+test_rank[1]+test_rank[2])/3, result.test_prediction)
    assert_array_almost_equal((oof_rank[0]+oof_rank[1]+oof_rank[2])/3, result.oof_prediction)
    assert result.score is None


def test_averaging_opt_maximize():
    X, y = make_classification_df(n_samples=1024)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    oof, test = _make_1st_stage_preds(X_train, y_train, X_test)

    best_single_model = max(roc_auc_score(y_train, oof[0]),
                            roc_auc_score(y_train, oof[1]),
                            roc_auc_score(y_train, oof[2]))

    result = averaging_opt(test, oof, y_train, roc_auc_score, higher_is_better=True)

    assert result.score >= best_single_model

    result_simple_avg = averaging(test, oof, y_train, eval_func=roc_auc_score)

    assert result.score >= result_simple_avg.score


def test_averaging_opt_minimize():
    X, y = make_regression_df(n_samples=1024)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    oof, test = _make_1st_stage_preds(X_train, y_train, X_test)

    best_single_model = min(mean_squared_error(y_train, oof[0]),
                            mean_squared_error(y_train, oof[1]),
                            mean_squared_error(y_train, oof[2]))

    result = averaging_opt(test, oof, y_train, mean_squared_error, higher_is_better=False)

    assert result.score <= best_single_model

    result_simple_avg = averaging(test, oof, y_train, eval_func=mean_squared_error)

    assert result.score <= result_simple_avg.score


def test_rank_averaging_opt_maximize():
    X, y = make_classification_df(n_samples=1024)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    oof, test = _make_1st_stage_preds(X_train, y_train, X_test)

    best_single_model = max(roc_auc_score(y_train, oof[0]),
                            roc_auc_score(y_train, oof[1]),
                            roc_auc_score(y_train, oof[2]))

    result = averaging_opt(test, oof, y_train, roc_auc_score, higher_is_better=True, rank_averaging=True)

    assert result.score >= best_single_model

    result_simple_avg = averaging(test, oof, y_train, eval_func=roc_auc_score, rank_averaging=True)

    assert result.score >= result_simple_avg.score

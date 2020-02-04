from typing import List

import numpy as np

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split, KFold

from nyaggle.experiment import autoprep_gbdt
from nyaggle.testing import make_classification_df
from nyaggle.validation import cross_validate, Take


def test_cv_sklean_binary():
    X, y = make_classification(n_samples=1024, n_features=20, class_sep=0.98, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    model = RidgeClassifier(alpha=1.0)

    pred_oof, pred_test, scores, _ = cross_validate(model, X_train, y_train, X_test, cv=5, eval_func=roc_auc_score)

    assert len(scores) == 5 + 1
    assert scores[-1] >= 0.85  # overall auc
    assert roc_auc_score(y_train, pred_oof) == scores[-1]
    assert roc_auc_score(y_test, pred_test) >= 0.85  # test score


def test_cv_sklean_regression():
    X, y = make_regression(n_samples=1024, n_features=20, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    model = Ridge(alpha=1.0)

    pred_oof, pred_test, scores, _ = cross_validate(model, X_train, y_train, X_test, cv=5, eval_func=r2_score)

    print(scores)
    assert len(scores) == 5 + 1
    assert scores[-1] >= 0.95  # overall r2
    assert r2_score(y_train, pred_oof) == scores[-1]
    assert r2_score(y_test, pred_test) >= 0.95  # test r2


def test_cv_lgbm():
    X, y = make_classification(n_samples=1024, n_features=20, class_sep=0.98, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    models = [LGBMClassifier(n_estimators=300) for _ in range(5)]

    pred_oof, pred_test, scores, importance = cross_validate(models, X_train, y_train, X_test, cv=5,
                                                             eval_func=roc_auc_score,
                                                             fit_params={'early_stopping_rounds': 200})

    print(scores)
    assert len(scores) == 5 + 1
    assert scores[-1] >= 0.85  # overall roc_auc
    assert roc_auc_score(y_train, pred_oof) == scores[-1]
    assert roc_auc_score(y_test, pred_test) >= 0.85  # test roc_auc
    assert roc_auc_score(y, models[0].predict_proba(X)[:, 1]) >= 0.85  # make sure models are trained
    assert len(importance) == 5
    assert list(importance[0].columns) == ['feature', 'importance']
    assert len(importance[0]) == 20


def test_cv_lgbm_df():
    X, y = make_classification_df(n_samples=1024, n_num_features=20, n_cat_features=1, class_sep=0.98, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    models = [LGBMClassifier(n_estimators=300) for _ in range(5)]

    pred_oof, pred_test, scores, importance = cross_validate(models, X_train, y_train, X_test, cv=5,
                                                             eval_func=roc_auc_score)

    print(scores)
    assert len(scores) == 5 + 1
    assert scores[-1] >= 0.85  # overall roc_auc
    assert roc_auc_score(y_train, pred_oof) == scores[-1]
    assert roc_auc_score(y_test, pred_test) >= 0.85  # test roc_auc
    assert roc_auc_score(y_test, models[0].predict_proba(X_test)[:, 1]) >= 0.85  # make sure models are trained
    assert len(importance) == 5
    assert list(importance[0].columns) == ['feature', 'importance']
    assert len(importance[0]) == 20 + 1
    assert models[0].booster_.num_trees() < 300  # making sure early stopping worked


def test_cv_cat_df():
    X, y = make_classification_df(n_samples=1024, n_num_features=20, n_cat_features=1, class_sep=0.98, random_state=0)
    X, _ = autoprep_gbdt(CatBoostClassifier(), X, None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    models = [CatBoostClassifier(n_estimators=300) for _ in range(5)]

    pred_oof, pred_test, scores, importance = cross_validate(models, X_train, y_train, X_test, cv=5,
                                                             eval_func=roc_auc_score,
                                                             fit_params={'cat_features': ['cat_0']})

    print(scores)
    assert len(scores) == 5 + 1
    assert scores[-1] >= 0.85  # overall roc_auc
    assert roc_auc_score(y_train, pred_oof) == scores[-1]
    assert roc_auc_score(y_test, pred_test) >= 0.85  # test roc_auc
    assert roc_auc_score(y_test, models[0].predict_proba(X_test)[:, 1]) >= 0.85  # make sure models are trained
    assert len(importance) == 5
    assert list(importance[0].columns) == ['feature', 'importance']
    assert len(importance[0]) == 20 + 1
    assert models[0].tree_count_ < 300  # making sure early stopping worked


def test_cv_partial_evaluate():
    X, y = make_classification(n_samples=1024, n_features=20, class_sep=0.98, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    model = RidgeClassifier(alpha=1.0)

    n = 0

    def _fold_count(*args):
        nonlocal n
        n += 1

    cv = Take(2, KFold(5))

    pred_oof, pred_test, scores, _ = cross_validate(model, X_train, y_train, X_test, cv=cv, eval_func=roc_auc_score,
                                                    on_each_fold=_fold_count)

    assert len(scores) == 2 + 1
    assert scores[-1] >= 0.8  # overall auc
    assert n == 2


def test_fit_params_callback():
    X, y = make_classification(n_samples=1024, n_features=20, class_sep=0.98, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    models = [LGBMClassifier(n_estimators=300) for _ in range(5)]

    sample_weights = np.random.randint(1, 10, size=len(X_train))
    sample_weights = sample_weights / sample_weights.sum()

    def fit_params(n: int, train_index: List[int], valid_index: List[int]):
        return {
            'early_stopping_rounds': 100,
            'sample_weight': list(sample_weights[train_index]),
            'eval_sample_weight': [list(sample_weights[valid_index])]
        }

    result_w_weight = cross_validate(models, X_train, y_train, X_test, cv=5,
                                     eval_func=roc_auc_score, fit_params=fit_params)


    result_wo_weight = cross_validate(models, X_train, y_train, X_test, cv=5,
                                      eval_func=roc_auc_score, fit_params={'early_stopping_rounds': 50})

    assert result_w_weight.scores[-1] != result_wo_weight.scores[-1]

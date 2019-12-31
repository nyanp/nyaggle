import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score

from nyaggle.model.cv import cv
from nyaggle.testing import make_classification_df


def test_cv_sklean_binary():
    X, y = make_classification(n_samples=1024, n_features=20, class_sep=0.98, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    model = RidgeClassifier(alpha=1.0)

    pred_oof, pred_test, scores = cv(model, X_train, y_train, X_test, nfolds=5, eval=roc_auc_score)

    assert len(scores) == 5 + 1
    assert scores[-1] >= 0.85  # overall auc
    assert roc_auc_score(y_train, pred_oof) == scores[-1]
    assert roc_auc_score(y_test, pred_test) >= 0.85  # test score


def test_cv_sklean_regression():
    X, y = make_regression(n_samples=1024, n_features=20, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    model = Ridge(alpha=1.0)

    pred_oof, pred_test, scores = cv(model, X_train, y_train, X_test, nfolds=5, eval=r2_score)

    print(scores)
    assert len(scores) == 5 + 1
    assert scores[-1] >= 0.95  # overall r2
    assert r2_score(y_train, pred_oof) == scores[-1]
    assert r2_score(y_test, pred_test) >= 0.95  # test r2


def test_cv_lgbm():
    X, y = make_classification(n_samples=1024, n_features=20, class_sep=0.98, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    models = [LGBMClassifier(n_estimators=300) for _ in range(5)]

    pred_oof, pred_test, scores = cv(models, X_train, y_train, X_test, nfolds=5, eval=roc_auc_score,
                                     fit_params={'early_stopping_rounds': 200})

    print(scores)
    assert len(scores) == 5 + 1
    assert scores[-1] >= 0.85  # overall roc_auc
    assert roc_auc_score(y_train, pred_oof) == scores[-1]
    assert roc_auc_score(y_test, pred_test) >= 0.85  # test roc_auc
    assert roc_auc_score(y, models[0].predict_proba(X)[:, 1]) >= 0.85  # make sure models are trained


def test_cv_lgbm_df():
    X, y = make_classification_df(n_samples=1024, n_num_features=20, n_cat_features=1, class_sep=0.98, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    models = [LGBMClassifier(n_estimators=300) for _ in range(5)]

    pred_oof, pred_test, scores = cv(models, X_train, y_train, X_test, nfolds=5, eval=roc_auc_score,
                                     fit_params={'early_stopping_rounds': 200})

    print(scores)
    assert len(scores) == 5 + 1
    assert scores[-1] >= 0.85  # overall roc_auc
    assert roc_auc_score(y_train, pred_oof) == scores[-1]
    assert roc_auc_score(y_test, pred_test) >= 0.85  # test roc_auc
    assert roc_auc_score(y_test, models[0].predict_proba(X_test)[:, 1]) >= 0.85  # make sure models are trained
import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.metrics import roc_auc_score, r2_score

from nyaggle.model.cv import cv


def test_cv_sklean_binary():
    X, y = make_classification(n_samples=1024, n_features=20, class_sep=0.98, random_state=0)

    model = RidgeClassifier(alpha=1.0)

    pred_oof, pred_test, scores = cv(model, X[:512, :], y[:512], X[512:, :], nfolds=5, eval=roc_auc_score)

    assert len(scores) == 5 + 1
    assert scores[-1] >= 0.85  # overall auc
    assert roc_auc_score(y[:512], pred_oof) == scores[-1]
    assert roc_auc_score(y[512:], pred_test) >= 0.85  # test score


def test_cv_sklean_regression():
    X, y = make_regression(n_samples=1024, n_features=20, random_state=0)

    model = Ridge(alpha=1.0)

    pred_oof, pred_test, scores = cv(model, X[:512, :], y[:512], X[512:, :], nfolds=5, eval=r2_score)

    print(scores)
    assert len(scores) == 5 + 1
    assert scores[-1] >= 0.95  # overall r2
    assert r2_score(y[:512], pred_oof) == scores[-1]
    assert r2_score(y[512:], pred_test) >= 0.95  # test r2


def test_cv_lgbm():
    X, y = make_classification(n_samples=1024, n_features=20, class_sep=0.98, random_state=0)

    models = [LGBMClassifier(n_estimators=300) for _ in range(5)]

    pred_oof, pred_test, scores = cv(models, X[:512, :], y[:512], X[512:, :], nfolds=5, eval=roc_auc_score,
                                     # additional parameters are passed to LGBMClassifier.fit()
                                     categorical_feature=[])

    print(scores)
    assert len(scores) == 5 + 1
    assert scores[-1] >= 0.85  # overall roc_auc
    assert roc_auc_score(y[:512], pred_oof) == scores[-1]
    assert roc_auc_score(y[512:], pred_test) >= 0.85  # test roc_auc
    assert roc_auc_score(y, models[0].predict_proba(X)[:, 1]) >= 0.85  # make sure models are trained


def test_cv_lgbm_df():
    X, y = make_classification(n_samples=1024, n_features=20, class_sep=0.98, random_state=0)
    cols = ['col{}'.format(i) for i in range(20)]

    X_train = pd.DataFrame(X[:512, :], columns=cols)
    y_train = pd.Series(y[:512], name='target')
    X_test = pd.DataFrame(X[512:, :], columns=cols)
    y_test = pd.Series(y[512:], name='target')

    X_train['cat'] = pd.Series(np.random.choice(['A', 'B', 'C'], size=512)).astype('category')
    X_test['cat'] = pd.Series(np.random.choice(['A', 'B', 'C'], size=512)).astype('category')

    models = [LGBMClassifier(n_estimators=300) for _ in range(5)]

    pred_oof, pred_test, scores = cv(models, X_train, y_train, X_test, nfolds=5, eval=roc_auc_score,
                                     categorical_feature=[])

    print(scores)
    assert len(scores) == 5 + 1
    assert scores[-1] >= 0.85  # overall roc_auc
    assert roc_auc_score(y[:512], pred_oof) == scores[-1]
    assert roc_auc_score(y[512:], pred_test) >= 0.85  # test roc_auc
    assert roc_auc_score(y_test, models[0].predict_proba(X_test)[:, 1]) >= 0.85  # make sure models are trained

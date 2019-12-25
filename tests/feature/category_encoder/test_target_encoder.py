import copy
import pytest

import numpy.testing as npt
import pandas as pd
from pandas.testing import assert_frame_equal

from sklearn.model_selection import KFold
import category_encoders as ce

from nyaggle.feature.category_encoder import TargetEncoder


def _test_target_encoder(X_train, y_train, X_test, **kw):


    cv = KFold(n_splits=2, random_state=42, shuffle=True)

    te = TargetEncoder(cv.split(X_train), **kw)

    ret_train = te.fit_transform(X_train, y_train)
    ret_test = te.transform(X_test)

    ret_train2 = copy.deepcopy(X_train)
    ret_test2 = copy.deepcopy(X_test)

    for train_idx, test_idx in cv.split(X_train):
        te2 = ce.TargetEncoder(**kw)

        if isinstance(X_train, pd.DataFrame):
            te2.fit(X_train.loc[train_idx, :], y_train.loc[train_idx])
            ret_train2.loc[test_idx] = te2.transform(ret_train2.loc[test_idx])
        else:
            te2.fit(X_train[train_idx, :], y_train[train_idx])
            ret_train2[test_idx] = te2.transform(ret_train2[test_idx])

    ret_train2 = ret_train2.astype(float)

    if isinstance(ret_train, pd.DataFrame):
        assert_frame_equal(ret_train, ret_train2)
    else:
        npt.assert_array_equal(ret_train, ret_train2)

    te2 = ce.TargetEncoder(**kw)
    te2.fit(X_train, y_train)

    ret_test2 = te2.transform(ret_test2)

    if isinstance(ret_train, pd.DataFrame):
        assert_frame_equal(ret_test, ret_test2)
    else:
        npt.assert_array_equal(ret_test, ret_test2)


def test_target_encoder():
    X_train = pd.DataFrame({
        'x': ['A', 'A', 'A', 'B', 'B', 'C'],

    })
    y_train = pd.Series([0, 0, 1, 0, 1, 1])
    X_test = pd.DataFrame({
        'x': ['A', 'B', 'C', 'D']
    })

    _test_target_encoder(X_train, y_train, X_test)


def test_target_encoder_ndarray():
    X_train = pd.DataFrame({
        'x': ['A', 'A', 'A', 'B', 'B', 'C'],

    })
    y_train = pd.Series([0, 0, 1, 0, 1, 1])
    X_test = pd.DataFrame({
        'x': ['A', 'B', 'C', 'D']
    })

    _test_target_encoder(X_train.values, y_train.values, X_test.values, return_df=False)

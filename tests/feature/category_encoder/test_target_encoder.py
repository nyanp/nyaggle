import pytest

import numpy.testing as npt
import pandas as pd
from pandas.testing import assert_frame_equal

from sklearn.model_selection import KFold
import category_encoders as ce

from nyaggle.feature.category_encoder import TargetEncoder


def test_target_encoder():
    df_train = pd.DataFrame({
        'x': ['A', 'A', 'A', 'B', 'B', 'C'],
        'y': [0, 0, 1, 0, 1, 1]
    })
    df_test = pd.DataFrame({
        'x': ['A', 'B', 'C', 'D'],
        'y': [None, None, None, None]
    })

    cv = KFold(n_splits=2, random_state=42, shuffle=True)

    te = TargetEncoder(cv.split(df_train), min_samples_leaf=1, smoothing=1)

    ret_train = te.fit_transform(df_train[['x']], df_train['y'])
    ret_test = te.transform(df_test[['x']])

    ret_train2 = df_train[['x']].copy()
    ret_test2 = df_test[['x']].copy()

    for train_idx, test_idx in cv.split(df_train):
        te2 = ce.TargetEncoder(min_samples_leaf=1, smoothing=1)
        te2.fit(df_train.loc[train_idx, 'x'], df_train['y'].loc[train_idx])
        ret_train2.loc[test_idx] = te2.transform(ret_train2.loc[test_idx])

    ret_train2 = ret_train2.astype(float)

    assert_frame_equal(ret_train, ret_train2)

    te2 = ce.TargetEncoder(min_samples_leaf=1, smoothing=1)
    te2.fit(df_train[['x']], df_train['y'])

    ret_test2 = te2.transform(ret_test2)

    assert_frame_equal(ret_test, ret_test2)

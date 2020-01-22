import os

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import nyaggle.feature_store as fs

from nyaggle.testing import get_temp_directory


def test_save_feature():
    df = pd.DataFrame()

    df['a'] = np.arange(100)

    with get_temp_directory() as tmp:
        fs.save_feature(df, 0, tmp)

        assert os.path.exists(os.path.join(tmp, '0.f'))


def test_load_feature():
    df = pd.DataFrame()

    df['a'] = np.arange(100)

    with get_temp_directory() as tmp:
        fs.save_feature(df, 0, tmp)

        df_loaded = fs.load_feature(0, tmp)
        assert_frame_equal(df, df_loaded)


def test_multi_columns():
    df = pd.DataFrame()

    df['a'] = np.arange(100)
    df['b'] = None

    with get_temp_directory() as tmp:
        fs.save_feature(df, 0, tmp)

        df_loaded = fs.load_feature(0, tmp)
        assert_frame_equal(df, df_loaded)


def test_various_dtypes():
    df = pd.DataFrame()

    df['a'] = np.arange(100).astype(float)
    df['b'] = np.arange(100).astype(int)
    df['c'] = np.arange(100).astype(np.uint8)
    df['d'] = np.arange(100).astype(np.uint16)
    df['e'] = np.arange(100).astype(np.uint32)
    df['f'] = np.arange(100).astype(np.int8)
    df['g'] = np.arange(100).astype(np.int16)
    df['h'] = np.arange(100).astype(np.int32)
    df['i'] = np.arange(100).astype(np.int64)

    with get_temp_directory() as tmp:
        fs.save_feature(df, 0, tmp)

        df_loaded = fs.load_feature(0, tmp)
        assert_frame_equal(df, df_loaded)


def test_load_features():
    df = pd.DataFrame()

    df['a'] = np.arange(100).astype(float)
    df['b'] = np.arange(100).astype(int)
    df['c'] = np.arange(100).astype(int)

    with get_temp_directory() as tmp:
        fs.save_feature(df[['b']], 0, tmp)
        fs.save_feature(df[['c']], 1, tmp)

        df_loaded = fs.load_features(df[['a']], [0, 1], tmp)
        assert_frame_equal(df, df_loaded)

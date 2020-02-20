from datetime import datetime

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import nyaggle.validation.split as split


def _random_uniform_dates(start_date: str, n_days: int, size: int):
    return pd.to_datetime(start_date) + pd.to_timedelta(np.random.randint(0, n_days, size=size), 'd')


def test_take():
    df = pd.DataFrame()
    df['id'] = np.arange(10)

    folds = split.Take(2, KFold(5)).split(df)

    train_index, test_index = next(folds)
    assert np.array_equal(test_index, np.array([0, 1]))

    train_index, test_index = next(folds)
    assert np.array_equal(test_index, np.array([2, 3]))

    with pytest.raises(StopIteration):
        next(folds)


def test_take_over():
    df = pd.DataFrame()
    df['id'] = np.arange(10)

    # k > base_validator.n_splits
    folds = split.Take(3, KFold(2)).split(df)

    train_index, test_index = next(folds)
    assert np.array_equal(test_index, np.array([0, 1, 2, 3, 4]))

    train_index, test_index = next(folds)
    assert np.array_equal(test_index, np.array([5, 6, 7, 8, 9]))

    with pytest.raises(StopIteration):
        next(folds)


def test_skip():
    df = pd.DataFrame()
    df['id'] = np.arange(10)

    kf = split.Skip(2, KFold(5))
    folds = kf.split(df)

    assert kf.get_n_splits() == 3

    train_index, test_index = next(folds)
    assert np.array_equal(test_index, np.array([4, 5]))

    train_index, test_index = next(folds)
    assert np.array_equal(test_index, np.array([6, 7]))

    train_index, test_index = next(folds)
    assert np.array_equal(test_index, np.array([8, 9]))

    with pytest.raises(StopIteration):
        next(folds)


def test_nth():
    df = pd.DataFrame()
    df['id'] = np.arange(10)

    kf = split.Nth(3, KFold(5))
    folds = kf.split(df)

    assert kf.get_n_splits() == 1

    train_index, test_index = next(folds)
    assert np.array_equal(test_index, np.array([4, 5]))

    with pytest.raises(StopIteration):
        next(folds)

    kf = split.Nth(1, KFold(5))
    folds = kf.split(df)

    assert kf.get_n_splits() == 1

    train_index, test_index = next(folds)
    assert np.array_equal(test_index, np.array([0, 1]))

    with pytest.raises(StopIteration):
        next(folds)


def test_time_series_split():
    df = pd.DataFrame()
    df['time'] = pd.date_range(start='2018/1/1', periods=5)

    folds = split.TimeSeriesSplit('time',
                                  [(('2018-01-01', '2018-01-02'), ('2018-01-02', '2018-01-04')),
                                   (('2018-01-02', '2018-01-03'), ('2018-01-04', '2018-01-06'))])

    assert folds.get_n_splits() == 2

    splits = folds.split(df)

    train_index, test_index = next(splits)
    assert np.array_equal(train_index, np.array([0]))
    assert np.array_equal(test_index, np.array([1, 2]))

    train_index, test_index = next(splits)
    assert np.array_equal(train_index, np.array([1]))
    assert np.array_equal(test_index, np.array([3, 4]))

    with pytest.raises(StopIteration):
        next(splits)


def test_time_series_open_range():
    df = pd.DataFrame()
    df['x'] = [1, 2, 3, 4, 5]
    df['time'] = pd.date_range(start='2018/1/1', periods=5)

    folds = split.TimeSeriesSplit(df['time'],
                                  [((None, '2018-01-03'), ('2018-01-03', None))])
    splits = folds.split(df)

    train_index, test_index = next(splits)
    assert np.array_equal(train_index, np.array([0, 1]))
    assert np.array_equal(test_index, np.array([2, 3, 4]))


def test_time_series_add_folds():
    df = pd.DataFrame()
    df['x'] = [1, 2, 3, 4, 5]
    df['time'] = pd.date_range(start='2018/1/1', periods=5)

    folds = split.TimeSeriesSplit(df['time'])

    assert folds.get_n_splits() == 0

    folds.add_fold((None, '2018-01-03'), ('2018-01-03', None))

    assert folds.get_n_splits() == 1


def test_sliding_window_split():
    window = split.SlidingWindowSplit('time',
                                      train_from='2018-01-20',
                                      train_to='2018-01-23',
                                      test_from='2018-01-27',
                                      test_to='2018-01-31',
                                      n_windows=3,
                                      stride=pd.to_timedelta(2, 'd'))

    #         train           test
    #  fold1: 01/16 - 01/19   01/23 - 01/27  (backtest 2)
    #  fold2: 01/18 - 01/21   01/25 - 01/29  (backtest 1)
    #  fold3: 01/20 - 01/23   01/27 - 01/31  (base window)

    expected = [
        ((datetime(2018, 1, 16), datetime(2018, 1, 19)), (datetime(2018, 1, 23), datetime(2018, 1, 27))),
        ((datetime(2018, 1, 18), datetime(2018, 1, 21)), (datetime(2018, 1, 25), datetime(2018, 1, 29))),
        ((datetime(2018, 1, 20), datetime(2018, 1, 23)), (datetime(2018, 1, 27), datetime(2018, 1, 31)))
    ]

    assert window.times == expected


def test_stratified_group_kfold_one_class_per_grp():
    sgf = split.StratifiedGroupKFold(2, shuffle=False)

    df = pd.DataFrame()
    df['group'] = [1, 1, 2, 2, 3, 3, 4, 4]
    df['y'] = [0, 0, 1, 1, 0, 0, 1, 1]
    df['x'] = [0, 1, 2, 3, 4, 5, 6, 7]

    assert sgf.get_n_splits(df, df['y'], df['group']) == 2

    splits = sgf.split(df, df['y'], df['group'])

    train_index, test_index = next(splits)
    assert np.array_equal(train_index, np.array([2, 3, 4, 5]))
    assert np.array_equal(test_index, np.array([0, 1, 6, 7]))

    train_index, test_index = next(splits)
    assert np.array_equal(train_index, np.array([0, 1, 6, 7]))
    assert np.array_equal(test_index, np.array([2, 3, 4, 5]))


def test_stratified_group_kfold_multi_class_per_fold():
    sgf = split.StratifiedGroupKFold(2, shuffle=False)

    df = pd.DataFrame()
    df['group'] = [1, 1, 2, 2, 3, 3, 4, 4]
    df['y'] = [0, 1, 0, 1, 1, 1, 1, 1]
    df['x'] = [0, 1, 2, 3, 4, 5, 6, 7]

    assert sgf.get_n_splits(df, df['y'], df['group']) == 2

    splits = sgf.split(df, df['y'], df['group'])

    train_index, test_index = next(splits)
    assert np.array_equal(train_index, np.array([0, 1, 4, 5]))
    assert np.array_equal(test_index, np.array([2, 3, 6, 7]))

    train_index, test_index = next(splits)
    assert np.array_equal(train_index, np.array([2, 3, 6, 7]))
    assert np.array_equal(test_index, np.array([0, 1, 4, 5]))


def test_stratified_group_kfold_imbalanced_group():
    sgf = split.StratifiedGroupKFold(2, shuffle=False)

    df = pd.DataFrame()
    df['group'] = [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4]
    df['y'] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    df['x'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    assert sgf.get_n_splits(df, df['y'], df['group']) == 2

    splits = sgf.split(df, df['y'], df['group'])

    train_index, test_index = next(splits)
    assert np.array_equal(train_index, np.array([8, 9, 10, 11]))
    assert np.array_equal(test_index, np.array([0, 1, 2, 3, 4, 5, 6, 7]))

    train_index, test_index = next(splits)
    assert np.array_equal(train_index, np.array([0, 1, 2, 3, 4, 5, 6, 7]))
    assert np.array_equal(test_index, np.array([8, 9, 10, 11]))


def test_stratified_group_kfold_y_with_continuous():
    sgf = split.StratifiedGroupKFold(2, shuffle=False)
    np.random.seed(1)
    df = pd.DataFrame()
    df['group'] = [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4]
    df['y'] = np.random.rand(len(df['group']))
    df['x'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    with pytest.raises(ValueError):
        splits = sgf.split(df, df['y'], df['group'])
        next(splits)


def test_stratified_group_kfold_groups_set_none():
    sgf = split.StratifiedGroupKFold(2, shuffle=False)

    df = pd.DataFrame()
    df['group'] = [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4]
    df['y'] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    df['x'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    with pytest.raises(ValueError):
        splits = sgf.split(df, df['y'], groups=None)
        next(splits)

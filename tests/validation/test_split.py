from datetime import datetime

import pytest
import numpy as np
import pandas as pd
import nyaggle.validation.split as split


def _random_uniform_dates(start_date: str, n_days: int, size: int):
    return pd.to_datetime(start_date) + pd.to_timedelta(np.random.randint(0, n_days, size=size), 'd')


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

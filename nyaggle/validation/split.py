import numbers
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target


def check_cv(cv: Union[int, Iterable, BaseCrossValidator] = 5,
             y: Optional[Union[pd.Series, np.ndarray]] = None,
             stratified: bool = False,
             random_state: int = 0):
    if cv is None:
        cv = 5
    if isinstance(cv, numbers.Integral):
        if stratified and (y is not None) and (type_of_target(y) in ('binary', 'multiclass')):
            return StratifiedKFold(cv, shuffle=True, random_state=random_state)
        else:
            return KFold(cv, shuffle=True, random_state=random_state)

    return model_selection.check_cv(cv, y, stratified)


class Take(BaseCrossValidator):
    """ Returns the first N folds of the base validator

    This validator wraps the base validator to take first n folds.

    Args:
        n:
            The number of folds.
        base_validator:
            The base validator to be wrapped.
    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.model_selection import KFold
        >>> from nyaggle.validation import Take

        >>> # take the first 3 folds out of 5
        >>> split = Take(3, KFold(5))
        >>> folds.get_n_splits()
        3
    """
    def __init__(self, n: int, base_validator: BaseCrossValidator):
        self.base_validator = base_validator
        self.n = n

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Args:
            X:
                Training data.
            y:
                Target.
            groups:
                Group indices.

        Yields:
            The training set and the testing set indices for that split.
        """
        generator = self.base_validator.split(X, y, groups)
        for i in range(min(self.n, self.base_validator.get_n_splits(X, y, groups))):
            yield next(generator)


class Skip(BaseCrossValidator):
    """ Skips the first N folds and returns the remaining folds

    This validator wraps the base validator to skip first n folds.

    Args:
        n:
            The number of folds to be skipped.
        base_validator:
            The base validator to be wrapped.
    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.model_selection import KFold
        >>> from nyaggle.validation import Skip

        >>> # take the last 2 folds out of 5
        >>> split = Skip(3, KFold(5))
        >>> folds.get_n_splits()
        2
    """
    def __init__(self, n: int, base_validator: BaseCrossValidator):
        self.base_validator = base_validator
        self.n = n

    def get_n_splits(self, X=None, y=None, groups=None):
        return max(self.base_validator.get_n_splits(X, y, groups) - self.n, 0)

    def split(self, X, y=None, groups=None):
        generator = self.base_validator.split(X, y, groups)

        for i in range(self.n):
            next(generator)

        for i in range(self.get_n_splits(X, y, groups)):
            yield next(generator)


class Nth(BaseCrossValidator):
    """ Returns N-th fold of the base validator

    This validator wraps the base validator to take n-th (1-origin) fold.

    Args:
        n:
            The number of folds to be taken.
        base_validator:
            The base validator to be wrapped.
    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.model_selection import KFold
        >>> from nyaggle.validation import Nth

        >>> # take the 3rd fold
        >>> split = Nth(3, KFold(5))
        >>> folds.get_n_splits()
        1
    """
    def __init__(self, n: int, base_validator: BaseCrossValidator):
        assert n > 0, "n is 1-origin and should be greater than 0"
        self.base_validator = Take(1, Skip(n-1, base_validator))
        self.n = n

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        generator = self.base_validator.split(X, y, groups)
        yield next(generator)


class TimeSeriesSplit(BaseCrossValidator):
    """ Time Series cross-validator

    Time Series cross-validator which provides train/test indices to split variable interval time series data.
    This class provides low-level API for time series validation strategy.
    This class is compatible with sklearn's ``BaseCrossValidator`` (base class of ``KFold``, ``GroupKFold`` etc).

    Args:
        source:
            The column name or series of timestamp.
        times:
            Splitting window, where times[i][0] and times[i][1] denotes train and test time interval in (i-1)th fold
            respectively. Each time interval should be pair of datetime or str, and the validator generates indices
            of rows where timestamp is in the half-open interval [start, end).
            For example, if ``times[i][0] = ('2018-01-01', '2018-01-03')``, indices for (i-1)th training data
            will be rows where timestamp value meets ``2018-01-01 <= t < 2018-01-03``.

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from nyaggle.validation import TimeSeriesSplit
        >>> df = pd.DataFrame()
        >>> df['time'] = pd.date_range(start='2018/1/1', periods=5)

        >>> folds = TimeSeriesSplit('time',
        >>>                         [(('2018-01-01', '2018-01-02'), ('2018-01-02', '2018-01-04')),
        >>>                          (('2018-01-02', '2018-01-03'), ('2018-01-04', '2018-01-06'))])

        >>> folds.get_n_splits()
        2

        >>> splits = folds.split(df)

        >>> train_index, test_index = next(splits)
        >>> train_index
        [0]
        >>> test_index
        [1, 2]

        >>> train_index, test_index = next(splits)
        >>> train_index
        [1]
        >>> test_index
        [3, 4]
    """
    datepair = Tuple[Union[datetime, str], Union[datetime, str]]

    def __init__(self, source: Union[pd.Series, str],
                 times: List[Tuple[datepair, datepair]] = None):
        self.source = source
        self.times = []
        if times:
            for t in times:
                self.add_fold(t[0], t[1])

    def _to_datetime(self, time: Union[str, datetime]):
        return time if isinstance(time, datetime) else pd.to_datetime(time)

    def _to_datetime_tuple(self, time: datepair):
        return self._to_datetime(time[0]), self._to_datetime(time[1])

    def add_fold(self, train_interval: datepair, test_interval: datepair):
        """
        Append 1 split to the validator.

        Args:
            train_interval:
                start and end time of training data.
            test_interval:
                start and end time of test data.
        """
        train_interval = self._to_datetime_tuple(train_interval)
        test_interval = self._to_datetime_tuple(test_interval)
        assert train_interval[1], "train_interval[1] should not be None"
        assert test_interval[0], "test_interval[0] should not be None"

        assert (not train_interval[0]) or (train_interval[0] <= train_interval[1]), "train_interval[0] < train_interval[1]"
        assert (not test_interval[1]) or (test_interval[0] <= test_interval[1]), "test_interval[0] < test_interval[1]"

        self.times.append((train_interval, test_interval))

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.times)

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Args:
            X:
                Training data.
            y:
                Ignored.
            groups:
                Ignored.

        Yields:
            The training set and the testing set indices for that split.
        """
        ts = X[self.source] if isinstance(self.source, str) else self.source

        for train_interval, test_interval in self.times:
            train_mask = ts < train_interval[1]
            if train_interval[0]:
                train_mask = (train_interval[0] <= ts) & train_mask

            test_mask = test_interval[0] <= ts
            if test_interval[1]:
                test_mask = test_mask & (ts < test_interval[1])

            yield np.where(train_mask)[0], np.where(test_mask)[0]


class SlidingWindowSplit(TimeSeriesSplit):
    """ Sliding window time series cross-validator

    Time Series cross-validator which provides train/test indices based on the sliding window to split
    variable interval time series data.
    Splitting for each fold will be as follows:

    .. code-block:: none

      Folds  Training data                                      Testing data
      1      ((train_from-(N-1)*stride, train_to-(N-1)*stride), (test_from-(N-1)*stride, test_to-(N-1)*stride))
      ...    ...                                                ...
      N-1    ((train_from-stride,       train_to-stride),       (test_from-stride,       test_to-stride))
      N      ((train_from,              train_to),              (test_from,              test_to))

    This class is compatible with sklearn's ``BaseCrossValidator`` (base class of ``KFold``, ``GroupKFold`` etc).

    Args:
        source:
            The column name or series of timestamp.
        train_from:
            Start datetime for the training data in the base split.
        train_to:
            End datetime for the training data in the base split.
        test_from:
            Start datetime for the testing data in the base split.
        test_to:
            End datetime for the testing data in the base split.
        n_windows:
            The number of windows (or folds) in the validation.
        stride:
            Time delta between folds.
    """

    date_or_str = Union[datetime, str]

    def __init__(self, source: Union[pd.Series, str],
                 train_from: date_or_str,
                 train_to: date_or_str,
                 test_from: date_or_str,
                 test_to: date_or_str,
                 n_windows: int,
                 stride: timedelta):
        super().__init__(source)

        train_from = self._to_datetime(train_from)
        train_to = self._to_datetime(train_to)
        test_from = self._to_datetime(test_from)
        test_to = self._to_datetime(test_to)

        splits = []

        for i in range(n_windows):
            splits.append(((train_from, train_to), (test_from, test_to)))
            train_from -= stride
            train_to -= stride
            test_from -= stride
            test_to -= stride

        for split in reversed(splits):
            self.add_fold(*split)

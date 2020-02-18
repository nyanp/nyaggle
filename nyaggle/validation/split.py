# Original work of StratifiedGroupKFold:
# https://github.com/Erotemic/baseline-viame-2018/blob/master/fishnet/util/sklearn_helpers.py
# -----------------------------------------------------------------------------
# Copyright 2018 Jon Crall
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------


import numbers
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ubelt as ub
import sklearn.model_selection as model_selection
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection._split import _BaseKFold


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


class StratifiedGroupKFold(_BaseKFold):
    """ Stratified K-Folds cross-validator with grouping

    Provides train/test indices to split data in train/test sets.
    This cross-validation object is a variation of GroupKFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class.
    Read more in the :ref:`User Guide <cross_validation>`.

    Args:
        n_splits :
            Number of folds. Must be at least 2.

    Example:
        >>> rng = np.random.RandomState(0)
        >>> groups = [1, 1, 3, 4, 2, 2, 7, 8, 8]
        >>> y      = [1, 1, 1, 1, 2, 2, 2, 3, 3]
        >>> X = np.empty((len(y), 0))
        >>> self = StratifiedGroupKFold(random_state=rng)
        >>> skf_list = list(self.split(X=X, y=y, groups=groups))
        >>> print(ub.repr2(skf_list, with_dtype=False))
        [
            (np.array([2, 3, 4, 5, 6]), np.array([0, 1, 7, 8])),
            (np.array([0, 1, 2, 7, 8]), np.array([3, 4, 5, 6])),
            (np.array([0, 1, 3, 4, 5, 6, 7, 8]), np.array([2])),
        ]
    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(StratifiedGroupKFold, self).__init__(n_splits, shuffle, random_state)

    def _make_test_folds(self, X, y=None, groups=None):
        """
        Args:
            X (ndarray):  data
            y (ndarray):  labels(default = None)
            groups (None): (default = None)
        """
        n_splits = self.n_splits
        y = np.asarray(y)
        n_samples = y.shape[0]

        unique_y, y_inversed = np.unique(y, return_inverse=True)
        n_classes = max(unique_y) + 1
        group_to_idxs = ub.group_items(range(len(groups)), groups)
        # unique_groups = list(group_to_idxs.keys())
        group_idxs = list(group_to_idxs.values())
        grouped_y = [y.take(idxs) for idxs in group_idxs]
        grouped_y_counts = np.array([
            np.bincount(y_, minlength=n_classes) for y_ in grouped_y])

        target_freq = grouped_y_counts.sum(axis=0)
        target_ratio = target_freq / target_freq.sum()

        # Greedilly choose the split assignment that minimizes the local
        # * squared differences in target from actual frequencies
        # * and best equalizes the number of items per fold
        # Distribute groups with most members first
        split_freq = np.zeros((n_splits, n_classes))
        # split_ratios = split_freq / split_freq.sum(axis=1)
        split_ratios = np.ones(split_freq.shape) / split_freq.shape[1]
        split_diffs = ((split_freq - target_ratio) ** 2).sum(axis=1)
        sortx = np.argsort(grouped_y_counts.sum(axis=1))[::-1]
        grouped_splitx = []
        for count, group_idx in enumerate(sortx):
            group_freq = grouped_y_counts[group_idx]
            cand_freq = split_freq + group_freq
            cand_ratio = cand_freq / cand_freq.sum(axis=1)[:, None]
            cand_diffs = ((cand_ratio - target_ratio) ** 2).sum(axis=1)
            # Compute loss
            losses = []
            other_diffs = np.array([
                sum(split_diffs[x + 1:]) + sum(split_diffs[:x])
                for x in range(n_splits)
            ])
            # penalize unbalanced splits
            ratio_loss = other_diffs + cand_diffs
            # penalize heavy splits
            freq_loss = split_freq.sum(axis=1)
            denom = freq_loss.sum()
            if denom == 0:
                freq_loss = freq_loss * 0
            else:
                freq_loss = freq_loss / denom
            losses = ratio_loss + freq_loss
            #-------
            splitx = np.argmin(losses)
            split_freq[splitx] = cand_freq[splitx]
            split_ratios[splitx] = cand_ratio[splitx]
            split_diffs[splitx] = cand_diffs[splitx]
            grouped_splitx.append(splitx)

        test_folds = np.empty(n_samples, dtype=np.int)
        for group_idx, splitx in zip(sortx, grouped_splitx):
            idxs = group_idxs[group_idx]
            test_folds[idxs] = splitx

        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y, groups)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        """
        Generate indices to split data into training and test set.
        """
        from sklearn.utils.validation import check_array
        y = check_array(y, ensure_2d=False, dtype=None)
        return super(StratifiedGroupKFold, self).split(X, y, groups)

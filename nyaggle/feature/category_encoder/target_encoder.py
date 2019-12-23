from itertools import tee
from typing import List, Optional, Iterable, Union

import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold

from nyaggle.feature.base import BaseFeaturizer

NAN_INT = 7535805


class KFoldEncoderWrapper(BaseFeaturizer):
    """KFold Wrapper for Transformer Mix-In

    This class wraps sklearn's transformer (e.g. category_encoders.TargetEncoder), and call them K-fold manner.

    Args:
        base_transformer:
            Transformer object to be wrapped.
        split:
            KFold, StratifiedKFold or kf.split(X, y)
    """
    def __init__(self, base_transformer: BaseEstimator, split: Union[Iterable, KFold, StratifiedKFold]):
        self.split = split
        self.n_splits = self._get_n_splits()
        self.transformers = [clone(base_transformer) for _ in range(self.n_splits + 1)]

    def _get_n_splits(self) -> int:
        if isinstance(self.split, (KFold, StratifiedKFold)):
            return self.split.get_n_splits()
        self.split, split = tee(self.split)
        return len([0 for _ in split])

    def _fit_train(self, X: pd.DataFrame, y: Optional[pd.Series]):
        if y is None:
            X_ = self.transformers[-1].transform(X)
            return self._post_transform(X_)

        if isinstance(self.split, (KFold, StratifiedKFold)):
            self.split = self.split.split(X, y)

        self.split, split = tee(self.split)
        X_ = X.copy()

        for i, (train_index, test_index) in enumerate(split):
            self.transformers[i].fit(X.iloc[train_index], y.iloc[train_index])
            X_.iloc[test_index, :] = self.transformers[i].transform(X.iloc[test_index])
        self.transformers[-1].fit(X, y)

        return self._post_transform(self._post_fit(X_, y))

    def _post_fit(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return X

    def _post_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fit_transform(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._fit_train(X, None)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None, **fit_params) -> pd.DataFrame:
        assert len(X) == len(y)

        if y.isnull().sum() > 0:
            # y == null is regarded as test data
            X_ = X.copy()
            X_.loc[~y.isnull(), :] = self._fit_train(X[~y.isnull()], y[~y.isnull()])
            X_.loc[y.isnull(), :] = self._fit_train(X[y.isnull()], None)
        else:
            X_ = self._fit_train(X, y)

        return X_


class TargetEncoder(KFoldEncoderWrapper):
    """Target Encoder

    KFold version of [category_encoders.TargetEncoder](https://contrib.scikit-learn.org/categorical-encoding/targetencoder.html).

    Args:
        split:
            KFold, StratifiedKFold or kf.split(X, y)
        cols:
            a list of columns to encode, if None, all string columns will be encoded.
        drop_invariant:
            boolean for whether or not to drop columns with 0 variance.
        handle_missing:
            options are ‘error’, ‘return_nan’ and ‘value’, defaults to ‘value’, which returns the target mean.
        handle_unknown:
            options are ‘error’, ‘return_nan’ and ‘value’, defaults to ‘value’, which returns the target mean.
        min_samples_leaf:
            minimum samples to take category average into account.
        smoothing:
            smoothing effect to balance categorical average vs prior. Higher value means stronger regularization.
            The value must be strictly bigger than 0.
    """
    def __init__(self, split: Union[Iterable, KFold, StratifiedKFold], cols: List[str] = None,
                 drop_invariant: bool = False, handle_missing: str = 'value', handle_unknown: str = 'value',
                 min_samples_leaf: int = 1, smoothing: float = 1.0):
        e = ce.TargetEncoder(cols=cols, drop_invariant=drop_invariant, return_df=True,
                             handle_missing=handle_missing,
                             handle_unknown=handle_unknown,
                             min_samples_leaf=min_samples_leaf, smoothing=smoothing)

        super().__init__(e, split)

    def _post_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = self.transformers[0].cols
        for c in cols:
            X[c] = X[c].astype(float)
        return X

from itertools import tee
from typing import List, Optional, Iterable, Union

import numpy as np
import pandas as pd
import category_encoders as ce
from category_encoders.utils import convert_input, convert_input_vector
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold

from nyaggle.feature.base import BaseFeaturizer


class KFoldEncoderWrapper(BaseFeaturizer):
    """KFold Wrapper for sklearn like interface

    This class wraps sklearn's TransformerMixIn (object that has fit/transform/fit_transform methods),
    and call them K-fold manner.

    Args:
        base_transformer:
            Transformer object to be wrapped.
        split:
            KFold, StratifiedKFold or kf.split(X, y)
        return_same_type:
            If True, `transform` and `fit_transform` return the same type as X.
            If False, these APIs always return a numpy array, similar to sklearn's API.
    """
    def __init__(self, base_transformer: BaseEstimator,
                 split: Union[Iterable, KFold, StratifiedKFold], return_same_type: bool = True):
        self.split = split
        self.n_splits = self._get_n_splits()
        self.transformers = [clone(base_transformer) for _ in range(self.n_splits + 1)]
        self.return_same_type = return_same_type

    def _get_n_splits(self) -> int:
        if isinstance(self.split, (KFold, StratifiedKFold)):
            return self.split.get_n_splits()
        self.split, split = tee(self.split)
        return len([0 for _ in split])

    def _fit_train(self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params) -> pd.DataFrame:
        if y is None:
            X_ = self.transformers[-1].transform(X)
            return self._post_transform(X_)

        if isinstance(self.split, (KFold, StratifiedKFold)):
            self.split = self.split.split(X, y)

        self.split, split = tee(self.split)
        X_ = X.copy()

        for i, (train_index, test_index) in enumerate(split):
            self.transformers[i].fit(X.iloc[train_index], y.iloc[train_index], **fit_params)
            X_.iloc[test_index, :] = self.transformers[i].transform(X.iloc[test_index])
        self.transformers[-1].fit(X, y, **fit_params)

        return self._post_transform(self._post_fit(X_, y))

    def _post_fit(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return X

    def _post_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit models for each fold.

        :param X: Data
        :param y: Target
        """
        self.fit_transform(X, y)

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform X

        :param X: Data
        :return: Transformed version of X. It will be pd.DataFrame If X is `pd.DataFrame` and return_same_type is True.
        """
        is_pandas = isinstance(X, pd.DataFrame)
        X_ = self._fit_train(X, None)
        return X_ if self.return_same_type and is_pandas else X_.values

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: pd.Series = None, **fit_params) \
            -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit models for each fold, then transform X

        :param X: Data
        :param y: Target
        :param fit_params: Additional parameters passed to models
        :return: Transformed version of X. It will be pd.DataFrame If X is `pd.DataFrame` and return_same_type is True.
        """
        assert len(X) == len(y)

        is_pandas = isinstance(X, pd.DataFrame)
        X = convert_input(X)
        y = convert_input_vector(y, X.index)

        if y.isnull().sum() > 0:
            # y == null is regarded as test data
            X_ = X.copy()
            X_.loc[~y.isnull(), :] = self._fit_train(X[~y.isnull()], y[~y.isnull()], **fit_params)
            X_.loc[y.isnull(), :] = self._fit_train(X[y.isnull()], None, **fit_params)
        else:
            X_ = self._fit_train(X, y, **fit_params)

        return X_ if self.return_same_type and is_pandas else X_.values


class TargetEncoder(KFoldEncoderWrapper):
    """Target Encoder

    KFold version of category_encoders.TargetEncoder in https://contrib.scikit-learn.org/categorical-encoding/targetencoder.html.

    Args:
        split:
            KFold, StratifiedKFold or cross-validation generator which determines the cross-validation splitting strategy.
        cols:
            A list of columns to encode, if None, all string columns will be encoded.
        drop_invariant:
            Boolean for whether or not to drop columns with 0 variance.
        handle_missing:
            Options are ‘error’, ‘return_nan’ and ‘value’, defaults to ‘value’, which returns the target mean.
        handle_unknown:
            Options are ‘error’, ‘return_nan’ and ‘value’, defaults to ‘value’, which returns the target mean.
        min_samples_leaf:
            Minimum samples to take category average into account.
        smoothing:
            Smoothing effect to balance categorical average vs prior. Higher value means stronger regularization.
            The value must be strictly bigger than 0.
        return_same_type:
            If True, `transform` and `fit_transform` return the same type as X.
            If False, these APIs always return a numpy array, similar to sklearn's API.
    """
    def __init__(self, split: Union[Iterable, KFold, StratifiedKFold], cols: List[str] = None,
                 drop_invariant: bool = False, handle_missing: str = 'value', handle_unknown: str = 'value',
                 min_samples_leaf: int = 1, smoothing: float = 1.0, return_same_type: bool = True):
        e = ce.TargetEncoder(cols=cols, drop_invariant=drop_invariant, return_df=True,
                             handle_missing=handle_missing,
                             handle_unknown=handle_unknown,
                             min_samples_leaf=min_samples_leaf, smoothing=smoothing)

        super().__init__(e, split, return_same_type)

    def _post_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = self.transformers[0].cols
        for c in cols:
            X[c] = X[c].astype(float)
        return X

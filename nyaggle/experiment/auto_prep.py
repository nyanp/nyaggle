from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_categorical_dtype
from sklearn.preprocessing import LabelEncoder


def autoprep_gbdt(algorithm_type: str, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame],
                  categorical_feature_to_treat: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if categorical_feature_to_treat is None:
        categorical_feature_to_treat = [c for c in X_train.columns if X_train[c].dtype.name in ['object', 'category']]

    # LightGBM:
    # Can handle categorical dtype. Otherwise, int, float or bool is acceptable for categorical columns.
    # https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support
    #
    # CatBoost:
    # int, float, bool or str is acceptable for categorical columns. NaN should be filled.
    # https://catboost.ai/docs/concepts/faq.html#why-float-and-nan-values-are-forbidden-for-cat-features
    #
    # XGBoost:
    # All categorical column should be encoded beforehand.

    if algorithm_type == 'lgbm':
        # LightGBM can handle categorical dtype natively
        categorical_feature_to_treat = [c for c in categorical_feature_to_treat if not is_categorical_dtype(X_train[c])]

    if algorithm_type == 'cat' and len(categorical_feature_to_treat) > 0:
        X_train = X_train.copy()
        X_test = X_test.copy() if X_test is not None else X_train.iloc[:1, :].copy()  # dummy
        for c in categorical_feature_to_treat:
            X_train[c], X_test[c] = _fill_na_by_unique_value(X_train[c], X_test[c])

    if algorithm_type in ('lgbm', 'xgb') and len(categorical_feature_to_treat) > 0:
        assert X_test is not None, "X_test is required for XGBoost with categorical variables"
        X_train = X_train.copy()
        X_test = X_test.copy()

        for c in categorical_feature_to_treat:
            X_train[c], X_test[c] = _fill_na_by_unique_value(X_train[c], X_test[c])
            le = LabelEncoder()
            concat = np.concatenate([X_train[c].values, X_test[c].values])
            concat = le.fit_transform(concat)
            X_train[c] = concat[:len(X_train)]
            X_test[c] = concat[len(X_train):]

    return X_train, X_test


def _fill_na_by_unique_value(strain: pd.Series, stest: Optional[pd.Series]) -> Tuple[pd.Series, pd.Series]:
    if is_categorical_dtype(strain):
        return strain.cat.codes, stest.cat.codes
    elif is_integer_dtype(strain.dtype):
        fillval = min(strain.min(), stest.min()) - 1
        return strain.fillna(fillval), stest.fillna(fillval)
    else:
        return strain.astype(str), stest.astype(str)

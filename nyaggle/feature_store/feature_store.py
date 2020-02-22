import functools
import os
import pyarrow
import warnings
from typing import List, Optional, Union

import pandas as pd
from tqdm import tqdm


def validate_train_test_difference(train: pd.Series, test: pd.Series):
    # % of nulls
    if test.isnull().mean() == 1.0:
        print(UNDEFINED)
        raise RuntimeError('Error in feature {}: all values in test data is null'.format(train.name))


def validate_feature(df: pd.DataFrame, y: pd.Series):
    if len(y) < len(df):
        # assuming that the first part of the dataframe is train part
        train = df.iloc[:len(y), :]
        test = df.iloc[len(y):, :]
    else:
        train = df[~y.isnull()]
        test = df[y.isnull()]

    for c in df.columns:
        validate_train_test_difference(train[c], test[c])


def save_feature(df: pd.DataFrame, feature_name: Union[int, str], directory: str = './features/',
                 with_csv_dump: bool = False, create_directory: bool = True,
                 reference_target_variable: Optional[pd.Series] = None, overwrite: bool = False):
    """
    Save pandas dataframe as feather-format

    Args:
        df:
            The dataframe to be saved.
        feature_name:
            The name of the feature. The output file will be ``{feature_name}.f``.
        directory:
            The directory where the feature will be stored.
        with_csv_dump:
            If True, the first 1000 lines are dumped to csv file for debug.
        create_directory:
            If True, create directory if not exists.
        reference_target_variable:
            If not None, instant validation will be made on the feature.
        overwrite:
            If False and file already exists, RuntimeError will be raised.
    """
    if create_directory:
        os.makedirs(directory, exist_ok=True)

    if reference_target_variable is not None:
        validate_feature(df, reference_target_variable)

    path = os.path.join(directory, str(feature_name) + '.f')

    if not overwrite and os.path.exists(path):
        raise RuntimeError('File already exists')

    df.to_feather(path)

    if with_csv_dump:
        df.head(1000).to_csv(os.path.join(directory, str(feature_name) + '.csv'), index=False)


def load_feature(feature_name: Union[int, str], directory: str = './features/',
                 ignore_columns: List[str] = None) -> pd.DataFrame:
    """
    Load feature as pandas DataFrame.

    Args:
        feature_name:
            The name of the feature (used in ``save_feature``).
        directory:
            The directory where the feature is stored.
        ignore_columns:
            The list of columns that will be dropped from the loaded dataframe.
    Returns:
        The feature dataframe
    """
    path = os.path.join(directory, str(feature_name) + '.f')

    df = pd.read_feather(path)
    if ignore_columns:
        return df.drop([c for c in ignore_columns if c in df.columns], axis=1)
    else:
        return df


def load_features(base_df: Optional[pd.DataFrame],
                  feature_names: List[Union[int, str]], directory: str = './features/',
                  ignore_columns: List[str] = None, create_directory: bool = True,
                  rename_duplicate: bool = True) -> pd.DataFrame:
    """
    Load features and returns concatenated dataframe

    Args:
        base_df:
            The base dataframe. If not None, resulting dataframe will consist of base and loaded feature columns.
        feature_names:
            The list of feature names to be loaded.
        directory:
            The directory where the feature is stored.
        ignore_columns:
            The list of columns that will be dropped from the loaded dataframe.
        create_directory:
            If True, create directory if not exists.
        rename_duplicate:
            If True, duplicated column name will be renamed automatically (feature name will be used as suffix).
            If False, duplicated columns will be as-is.
    Returns:
        The merged dataframe
    """
    if create_directory:
        os.makedirs(directory, exist_ok=True)

    dfs = [load_feature(f, directory=directory, ignore_columns=ignore_columns) for f in tqdm(feature_names)]

    if base_df is None:
        base_df = dfs[0]
        dfs = dfs[1:]
        feature_names = feature_names[1:]

    columns = list(base_df.columns)

    for df, feature_name in zip(dfs, feature_names):
        if len(df) != len(base_df):
            raise RuntimeError('DataFrame length are different. feature={}'.format(feature_name))

        for c in df.columns:
            if c in columns:
                warnings.warn('A feature name {} is duplicated.'.format(c))

                if rename_duplicate:
                    while c in columns:
                        c += '_' + str(feature_name)
                    warnings.warn('The duplicated name in feature={} will be renamed to {}'.format(feature_name, c))
            columns.append(c)

    concatenated = pd.concat([base_df] + dfs, axis=1)
    concatenated.columns = columns
    return concatenated


def cached_feature(feature_name: Union[int, str], directory: str = './features/', ignore_columns: List[str] = None):
    """
    Decorator to wrap a function which returns pd.DataFrame with a memorizing callable that saves dataframe using
    ``feature_store.save_feature``.

    Args:
        feature_name:
            The name of the feature (used in ``save_feature``).
        directory:
            The directory where the feature is stored.
        ignore_columns:
            The list of columns that will be dropped from the loaded dataframe.

    Example:
        >>> from nyaggle.feature_store import cached_feature
        >>>
        >>> @cached_feature('x')
        >>> def make_feature_x(param) -> pd.DataFrame:
        >>>     print('called')
        >>>     ...
        >>>     return df
        >>>
        >>> x = make_feature_x(...)  # if x.f does not exist, call the function and save result to x.f
        "called"
        >>> x = make_feature_x(...)  # load from file in the second time
    """
    def _decorator(fun):
        @functools.wraps(fun)
        def _decorated_fun(*args, **kwargs):
            try:
                return load_feature(feature_name, directory, ignore_columns)
            except (pyarrow.ArrowIOError, IOError):
                df = fun(*args, **kwargs)
                assert isinstance(df, pd.DataFrame), "returning value of @cached_feature should be pd.DataFrame"
                save_feature(df, feature_name, directory)
                return df

        return _decorated_fun

    return _decorator

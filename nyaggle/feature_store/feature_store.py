import os
from typing import List, Union

from tqdm import tqdm
import pandas as pd


def save_feature(df: pd.DataFrame, feature_name: Union[int, str], directory: str = './features/',
                 with_csv_dump: bool = False, create_directory: bool = True):
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
    """
    if create_directory:
        os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, str(feature_name) + '.f')
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


def load_features(base_df: pd.DataFrame, feature_names: List[Union[int, str]], directory: str = './features/',
                  ignore_columns: List[str] = None, create_directory: bool = True) -> pd.DataFrame:
    """
    Load features and returns concatenated dataframe

    Args:
        base_df:
            The base dataframe.
        feature_names:
            The list of feature names to be loaded.
        directory:
            The directory where the feature is stored.
        ignore_columns:
            The list of columns that will be dropped from the loaded dataframe.
        create_directory:
            If True, create directory if not exists.
    Returns:
        The merged dataframe
    """
    if create_directory:
        os.makedirs(directory, exist_ok=True)

    dfs = [load_feature(f, directory=directory, ignore_columns=ignore_columns) for f in tqdm(feature_names)]

    for df, feature_name in zip(dfs, feature_names):
        if len(df) != len(base_df):
            raise RuntimeError('DataFrame length are different. feature_id: {}'.format(feature_name))

    return pd.concat([base_df] + dfs, axis=1)

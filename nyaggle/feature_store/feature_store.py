import os
from typing import List

from tqdm import tqdm
import pandas as pd


def save_feature(df: pd.DataFrame, feature_name: str, directory: str = './features/', with_csv_dump: bool = False):
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
    """
    path = os.path.join(directory, feature_name + '.f')
    df.to_feather(path)

    if with_csv_dump:
        df.head(1000).to_csv(os.path.join(directory, feature_name + '.csv'), index=False)


def load_feature(feature_name: str, directory: str = './features/') -> pd.DataFrame:
    """
    Load feature as pandas DataFrame.

    Args:
        feature_name:
            The name of the feature (used in ``save_feature``).
        directory:
            The directory where the feature is stored.
    Returns:
        The feature dataframe
    """
    path = os.path.join(directory, feature_name + '.f')

    return pd.read_feather(path)


def load_features(base_df: pd.DataFrame, feature_names: List[str], directory: str = './features/') -> pd.DataFrame:
    """
    Load features and returns concatenated dataframe

    Args:
        base_df:
            The base dataframe.
        feature_names:
            The list of feature names to be loaded.
        directory:
            The directory where the feature is stored.
    Returns:
        The merged dataframe
    """
    dfs = [load_feature(f, directory) for f in tqdm(feature_names)]

    for feature_name, df in zip(dfs, feature_names):
        if len(df) != base_df:
            raise RuntimeError('DataFrame length are different. feature_id: {}'.format(feature_name))

    return pd.concat([base_df] + dfs, axis=1)

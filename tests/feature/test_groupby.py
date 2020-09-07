import numpy as np
import pandas as pd
import pytest
from sklearn import datasets

from nyaggle.feature.groupby import aggregation


@pytest.fixture
def iris_dataframe():
    iris = datasets.load_iris()
    df = pd.DataFrame(np.concatenate([iris.data,
                                      iris.target.reshape((iris.target.shape[0], 1))], axis=1))
    df.columns = ['sl', 'sw', 'pl', 'pw', 'species']
    group_key = 'species'
    group_values = ['sl', 'sw', 'pl', 'pw']
    return df, group_key, group_values


def custom_function(x):
    return np.sum(x)


def test_return_type_by_aggregation(iris_dataframe):
    df, group_key, group_values = iris_dataframe
    agg_methods = ["max", np.sum, custom_function]
    new_df, new_cols = aggregation(df, group_key, group_values,
                                   agg_methods)
    assert isinstance(new_df, pd.DataFrame)
    assert isinstance(new_cols, list)


@pytest.mark.parametrize('agg_method', [[int], [lambda x: np.max(x)]])
def test_assert_by_aggregation(iris_dataframe, agg_method):
    df, group_key, group_values = iris_dataframe
    with pytest.raises(ValueError):
        aggregation(df, group_key, group_values, agg_method)

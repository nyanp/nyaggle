import pandas as pd
import pytest

from nyaggle.feature.groupby import aggregation


@pytest.fixture
def dataframe():
    df = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5],
         "b": ["a", "a", "a", "b", "b"],
         "c": [0, 0, 1, 1, 1],
         }
    )
    return df


def test_aggregation(dataframe):
    df = dataframe
    group_key = 'b'
    group_values = ["a", "c"]
    agg_methods = ["max"]
    new_df, new_cols = aggregation(df, group_key, group_values,
                                   agg_methods)

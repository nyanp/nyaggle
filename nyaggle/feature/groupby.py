# Modified work:
# -----------------------------------------------------------------------------
# Copyright (c) 2020 Kota Yuhara (@wakamezake)
# -----------------------------------------------------------------------------

# Original work of aggregation:
# https://github.com/pfnet-research/xfeat/blob/master/xfeat/helper.py
# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2020 Preferred Networks, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------


from types import FunctionType, LambdaType
from typing import Callable, List, Tuple, Union

import pandas as pd
from pandas.core.common import get_callable_name


def _is_lambda_function(obj):
    """
    Example:
        >>> import numpy as np
        >>> def custom_function(x): return np.sum(x)
        >>> _is_lambda_function(lambda x: np.sum(x))
        True
        >>> _is_lambda_function(np.sum)
        False
        >>> _is_lambda_function(custom_function)
        False
    """
    # It's worth noting that types.LambdaType is an alias for types.FunctionType
    return isinstance(obj, LambdaType) and obj.__name__ == "<lambda>"


def aggregation(
        input_df: pd.DataFrame,
        group_key: str,
        group_values: List[str],
        agg_methods: List[Union[str, FunctionType]],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate values after grouping table rows by a given key.

    Args:
        input_df:
            Input data frame.
        group_key:
            Used to determine the groups for the groupby.
        group_values:
            Used to aggregate values for the groupby.
        agg_methods:
            List of function or function names, e.g. ['mean', 'max', 'min', numpy.mean].
            Do not use a lambda function because the name attribute of the lambda function cannot generate a unique string of column names in <lambda>.
    Returns:
        Tuple of output dataframe and new column names.
    """
    new_df = input_df.copy()

    new_cols = []
    for agg_method in agg_methods:
        if _is_lambda_function(agg_method):
            raise ValueError('Not supported lambda function.')
        elif isinstance(agg_method, str):
            pass
        elif isinstance(agg_method, FunctionType):
            pass
        elif isinstance(agg_method, Callable):
            pass
        else:
            raise ValueError('Supported types are: {} or {}.'
                             ' Got {} instead.'.format(str, Callable, type(agg_method)))

    for agg_method in agg_methods:
        for col in group_values:
            # only str or FunctionType
            if isinstance(agg_method, str):
                agg_method_name = agg_method
            else:
                agg_method_name = get_callable_name(agg_method)
            new_col = "agg_{}_{}_by_{}".format(agg_method_name, col, group_key)

            df_agg = (
                input_df[[col] + [group_key]].groupby(group_key)[[col]].agg(
                    agg_method)
            )
            df_agg.columns = [new_col]
            new_cols.append(new_col)
            new_df = new_df.merge(
                df_agg, how="left", right_index=True, left_on=group_key
            )

    return new_df, new_cols

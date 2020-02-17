# safe_instance function is originally from:
# https://github.com/slundberg/shap/blob/master/shap/common.py

# The MIT License (MIT)
#
# Copyright (c) 2018 Scott Lundberg
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

import importlib
from typing import List, Tuple, Union


def is_instance(obj, class_path_str: Union[str, List, Tuple]) -> bool:
    """
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.
    Checks if obj is an instance of type specified by class_path_str.
    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`
    Returns
    --------
    bool: True if isinstance is true and the package exists, False otherwise
    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = ['']

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError("class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'")

        # Splits on last occurence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # Check module exists
        try:
            spec = importlib.util.find_spec(module_name)
        except:
            spec = None
        if spec is None:
            continue

        module = importlib.import_module(module_name)

        # Get class
        _class = getattr(module, class_name, None)
        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False


def is_gbdt_instance(obj, algorithm_type: Union[str, Tuple]) -> bool:
    if isinstance(algorithm_type, str):
        algorithm_type = (algorithm_type,)

    gbdt_instance_name = {
        'lgbm': 'lightgbm.sklearn.LGBMModel',
        'xgb': 'xgboost.sklearn.XGBModel',
        'cat': 'catboost.core.CatBoost'
    }

    return is_instance(obj, tuple(gbdt_instance_name[t] for t in algorithm_type))

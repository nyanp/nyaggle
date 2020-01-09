import numbers
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target


def check_cv(cv: Union[int, Iterable, KFold, StratifiedKFold] = 5,
             y: Optional[Union[pd.Series, np.ndarray]] = None,
             stratified: bool = False,
             random_state: int = 0):
    if cv is None:
        cv = 5
    if isinstance(cv, numbers.Integral):
        if stratified and (y is not None) and (type_of_target(y) in ('binary', 'multiclass')):
            return StratifiedKFold(cv, shuffle=True, random_state=random_state)
        else:
            return KFold(cv, shuffle=True, random_state=random_state)

    return model_selection.check_cv(cv, y, stratified)

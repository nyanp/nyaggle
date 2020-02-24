from typing import Optional

import numpy as np
import pandas as pd


def make_submission_df(test_prediction: np.ndarray, sample_submission: Optional[pd.DataFrame] = None,
                       y: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Make a dataframe formatted as a kaggle competition style.

    Args:
        test_prediction:
            A test prediction to be formatted.
        sample_submission:
            A sample dataframe alined with test data (Usually in Kaggle, it is available as sample_submission.csv).
            The submission file will be created with the same schema as this dataframe.
        y:
            Target variables which is used for inferring the column name. Ignored if ``sample_submission`` is passed.
    Returns:
        The formatted dataframe
    """
    if sample_submission is not None:
        submit_df = sample_submission.copy()

        if test_prediction.ndim > 1 and test_prediction.shape[1] > 1:
            n_id_cols = submit_df.shape[1] - test_prediction.shape[1]
            for i in range(test_prediction.shape[1]):
                submit_df.iloc[:, n_id_cols + i] = test_prediction[:, i]
        else:
            submit_df.iloc[:, -1] = test_prediction
    else:
        submit_df = pd.DataFrame()
        id_col_name = y.index.name if y is not None and y.index.name else 'id'

        submit_df[id_col_name] = np.arange(len(test_prediction))

        if test_prediction.ndim > 1 and test_prediction.shape[1] > 1:
            tgt_col_names = sorted(y.unique()) if y is not None else [str(i) for i in range(test_prediction.shape[1])]
            for i, y in enumerate(tgt_col_names):
                submit_df[y] = test_prediction[:, i]
        else:
            tgt_col_name = y.name if y is not None and y.name else 'target'
            submit_df[tgt_col_name] = test_prediction

    return submit_df

import glob
import os
from typing import List

import numpy as np
import pandas as pd


def average_results(source_files: List[str], output_filename: str, weight: List[float] = None,
                    input_format: str = 'csv', sample_submission_filename: str = None):
    """
    Calculate ensemble

    Args:
         source_files:
             Path of result file or experiment directory for each single model.
         output_filename:
             Output submission filename.
         weight:
             Weight for each model.
         input_format:
             'csv' or 'npy'
         sample_submission_filename:
             sample_submission
    :return: 
    """
    prediction = None

    if weight is None:
        weight = np.ones((len(source_files))) / len(source_files)

    for i, s in enumerate(source_files):
        if os.path.isdir(s):
            if input_format == 'csv':
                pattern = os.path.join(s, '*.csv')
            elif input_format == 'npy':
                pattern = os.path.join(s, 'test.npy')
            s = glob.glob(pattern)[0]

        assert os.path.exists(s), 'File not found: {}'.format(s)

        if input_format == 'csv':
            df = pd.read_csv(s)
            assert df.shape[1] == 2, 'Multiclass is not supported'
            v = df.values[:, 1]
        else:
            v = np.load(s)

        v *= weight[i]

        if prediction is None:
            prediction = v
        else:
            prediction += v

    if sample_submission_filename is not None:
        df = pd.read_csv(sample_submission_filename)
    else:
        if os.path.isdir(source_files[0]):
            dir = source_files[0]
        else:
            dir = os.path.dirname(source_files[0])
        path = glob.glob(os.path.join(dir, '*.csv'))[0]
        df = pd.read_csv(path)

    df[df.columns[-1]] = v
    df.to_csv(output_filename, index=False)

    return df


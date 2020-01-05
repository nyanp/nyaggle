import os
import uuid
from logging import getLogger, FileHandler, DEBUG, Logger
from typing import Optional

import numpy as np
import pandas as pd


class Experiment(object):
    """Minimal experiment logger for Kaggle

    This module provides minimal functionality for logging Kaggle experiments.
    The output files are laid out as follows:

    .. code-block:: none

      <logging_directory>/
          <log_filename>            <== output of log()
          <metrics_filename>        <== output of log_metrics(), format: name,score

    Args:
        logging_directory:
            Path to directory where output is stored.
        overwrite:
            If True, contents in ``logging_directory`` will be overwritten.
        log_filename:
            The name of debug log file created under logging_directory.
        metrics_filename:
            The name of score log file created under logging_directory.
        custom_logger:
            Custom logger to be used instead of default logger.
    """

    def __init__(self,
                 logging_directory: str,
                 overwrite: bool,
                 log_filename: str = 'log.txt',
                 metrics_filename: str = 'metrics.txt',
                 custom_logger: Optional[Logger] = None
                 ):
        os.makedirs(logging_directory, exist_ok=overwrite)
        self.logging_directory = logging_directory
        if custom_logger is not None:
            self.logger = custom_logger
            self.is_custom = True
        else:
            self.logger = getLogger(str(uuid.uuid4()))
            self.logger.addHandler(FileHandler(os.path.join(logging_directory, log_filename)))
            self.logger.setLevel(DEBUG)
            self.is_custom = False
        self.metrics = open(os.path.join(logging_directory, metrics_filename), mode='w')

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, trace):
        self.close()

    def log(self, text: str):
        self.logger.info(text)

    def log_metrics(self, name: str, score: float):
        self.metrics.write('{},{}\n'.format(name, score))
        self.metrics.flush()

    def log_numpy(self, name: str, array: np.ndarray):
        np.save(os.path.join(self.logging_directory, name), array)

    def get_logger(self):
        return self.logger

    def log_dataframe(self, name: str, df: pd.DataFrame, format: str = 'feather'):
        if format == 'feather':
            df.to_feather(os.path.join(self.logging_directory, name))
        elif format == 'csv':
            df.to_csv(os.path.join(self.logging_directory, name), index=False)
        else:
            raise RuntimeError('format not supported')

    def close(self):
        if not self.is_custom:
            for h in self.logger.handlers:
                h.close()
        self.metrics.close()


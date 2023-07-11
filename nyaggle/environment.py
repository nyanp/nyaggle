# pytorch

try:
    import torch  # noQA
    _has_torch = True
except ImportError:
    _has_torch = False


def requires_torch():
    if not _has_torch:
        raise ImportError('You need to install pytorch before using this API.')


# mlflow

try:
    import mlflow  # noQA
    _has_mlflow = True
except ImportError:
    _has_mlflow = False


def requires_mlflow():
    if not _has_mlflow:
        raise ImportError('You need to install mlflow before using this API.')


# lightgbm


try:
    import lightgbm  # noQA
    _has_lightgbm = True
except ImportError:
    _has_lightgbm = False


def requires_lightgbm():
    if not _has_lightgbm:
        raise ImportError('You need to install lightgbm before using this API.')


# lightgbm


try:
    import catboost  # noQA
    _has_catboost = True
    # TODO check catboost version >= 0.17
except ImportError:
    _has_catboost = False


def requires_catboost():
    if not _has_catboost:
        raise ImportError('You need to install catboost before using this API.')


# xgboost


try:
    import xgboost  # noQA
    _has_xgboost = True
except ImportError:
    _has_xgboost = False


def requires_xgboost():
    if not _has_xgboost:
        raise ImportError('You need to install xgboost before using this API.')

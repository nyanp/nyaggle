# pytorch

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False


def has_torch():
    return _has_torch


def requires_torch():
    if not has_torch():
        raise ImportError('You need to install pytorch before using this API.')


# mlflow

try:
    import mlflow
    _has_mlflow = True
except ImportError:
    _has_mlflow = False


def has_mlflow():
    return _has_mlflow


def requires_mlflow():
    if not has_mlflow():
        raise ImportError('You need to install mlflow before using this API.')

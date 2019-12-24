
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

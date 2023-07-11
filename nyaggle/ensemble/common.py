from collections import namedtuple

EnsembleResult = namedtuple('EnsembleResult', ['test_prediction', 'oof_prediction', 'score'])

from enum import Enum


class Language(Enum):
    EN = 1
    JP = 2


class PoolingStrategy(Enum):
    REDUCE_MEAN = 1,
    REDUCE_MAX = 2,
    REDUCE_MEAN_MAX = 3,
    CLS_TOKEN = 4

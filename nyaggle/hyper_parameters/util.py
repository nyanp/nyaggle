from more_itertools import first_true
from typing import Dict, List


def list_param_names(param_table: List[Dict]):
    return [p['name'] for p in param_table]


def get_hyperparam_byname(param_table: List[Dict], name: str, with_metadata: bool):
    found = first_true(param_table, pred=lambda x: x['name'] == name)
    if found is None:
        raise RuntimeError('Hyperparameter {} not found.'.format(name))

    if with_metadata:
        return found
    else:
        return found['parameters']

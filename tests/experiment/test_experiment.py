import json
import os
import pytest

import pandas as pd
import numpy as np

from nyaggle.experiment import Experiment


def test_log_params(tmpdir_name):
    with Experiment(tmpdir_name) as e:
        e.log_param('x', 1)
        e.log_param('x', 2)
        e.log_params({
            'y': 'ABC',
            'z': None,
        })

    with open(os.path.join(tmpdir_name, 'params.json'), 'r') as f:
        params = json.load(f)

        expected = {
            'x': 2,      # if the key is duplicated, the latter one is stored
            'y': 'ABC',
            'z': 'None'  # all non-numerical values are casted to string before logging
        }
        assert params == expected


def test_log_params_empty(tmpdir_name):
    with Experiment(tmpdir_name):
        pass

    with open(os.path.join(tmpdir_name, 'params.json'), 'r') as f:
        params = json.load(f)
        assert params == {}


def test_log_metrics(tmpdir_name):
    with Experiment(tmpdir_name) as e:
        e.log_metric('x', 1)
        e.log_metric('x', 2)
        e.log_metrics({
            'y': 3,
            'z': 4,
        })

    with open(os.path.join(tmpdir_name, 'metrics.json'), 'r') as f:
        params = json.load(f)

        expected = {
            'x': 2,
            'y': 3,
            'z': 4,
        }
        assert params == expected


def test_log_metrics_empty(tmpdir_name):
    with Experiment(tmpdir_name):
        pass

    with open(os.path.join(tmpdir_name, 'metrics.json'), 'r') as f:
        params = json.load(f)
        assert params == {}


def test_log_dict(tmpdir_name):
    with Experiment(tmpdir_name) as e:
        e.log_dict('foo', {'a': 1, 'b': 'foo', 'c': {'d': 'e', 'f': {}, 'g': {'h': 'i'}}})

    with open(os.path.join(tmpdir_name, 'params.json'), 'r') as f:
        params = json.load(f)
        assert params == {
            'foo.a': 1,
            'foo.b': 'foo',
            'foo.c.d': 'e',
            'foo.c.f': '{}',
            'foo.c.g.h': 'i'
        }


def test_error_while_experiment(tmpdir_name):
    try:
        with Experiment(tmpdir_name) as e:
            e.log_metric('x', 0.5)
            e.log_param('foo', 'bar')
            e.log_numpy('np', np.zeros(100))
            e.log_dataframe('df', pd.DataFrame({'a': [1, 2, 3]}))

        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    # all logs are saved even if error raised inside experiment
    with open(os.path.join(tmpdir_name, 'metrics.json'), 'r') as f:
        metrics = json.load(f)
        assert metrics == {'x': 0.5}

    with open(os.path.join(tmpdir_name, 'params.json'), 'r') as f:
        params = json.load(f)
        assert params == {'foo': 'bar'}

    assert os.path.exists(os.path.join(tmpdir_name, 'np.npy'))
    assert os.path.exists(os.path.join(tmpdir_name, 'df.f'))


def test_experiment_duplicated_error(tmpdir_name):
    with Experiment(tmpdir_name) as e:
        e.log_metric('CV', 0.97)

    with pytest.raises(ValueError):
        with Experiment(tmpdir_name):
            pass

    with pytest.raises(ValueError):
        with Experiment(tmpdir_name, if_exists='error'):
            pass


def test_experiment_duplicated_replace(tmpdir_name):
    with Experiment(tmpdir_name) as e:
        e.log_metric('CV', 0.97)

    with Experiment(tmpdir_name, if_exists='replace') as e:
        e.log_metric('LB', 0.95)

    with open(os.path.join(tmpdir_name, 'metrics.json')) as f:
        metrics = json.load(f)

        # replaced by the new result
        assert 'LB' in metrics
        assert 'CV' not in metrics


def test_experiment_duplicated_append(tmpdir_name):
    with Experiment(tmpdir_name) as e:
        e.log_metric('CV', 0.97)

    with Experiment(tmpdir_name, if_exists='append') as e:
        e.log_metric('LB', 0.95)

    with open(os.path.join(tmpdir_name, 'metrics.json')) as f:
        metrics = json.load(f)

        # appended to the existing result
        assert 'LB' in metrics
        assert 'CV' in metrics


def test_experiment_duplicated_rename(tmpdir_name):
    with Experiment(tmpdir_name) as e:
        e.log_metric('CV', 0.97)

    with Experiment(tmpdir_name, if_exists='rename') as e:
        e.log_metric('LB', 0.95)

    with open(os.path.join(tmpdir_name, 'metrics.json')) as f:
        metrics = json.load(f)
        assert 'LB' not in metrics
        assert 'CV' in metrics

    with open(os.path.join(tmpdir_name + '_1', 'metrics.json')) as f:
        metrics = json.load(f)
        assert 'LB' in metrics
        assert 'CV' not in metrics


def test_experiment_duplicated_replace_mlflow(tmpdir_name):
    import mlflow

    with Experiment(tmpdir_name, with_mlflow=True) as e:
        e.log_metric('CV', 0.97)
        run_id_old = e.mlflow_run_id

    with Experiment(tmpdir_name, with_mlflow=True, if_exists='replace') as e:
        e.log_metric('LB', 0.95)
        run_id_new = e.mlflow_run_id

    assert run_id_old != run_id_new

    client = mlflow.tracking.MlflowClient()
    old_run = client.get_run(run_id_old)
    new_run = client.get_run(run_id_new)
    assert old_run.info.lifecycle_stage == 'deleted'
    assert new_run.info.lifecycle_stage == 'active'


def test_experiment_duplicated_append_mlflow(tmpdir_name):
    with Experiment(tmpdir_name, with_mlflow=True) as e:
        e.log_metric('CV', 0.97)
        run_id_old = e.mlflow_run_id

    with Experiment(tmpdir_name, with_mlflow=True, if_exists='append') as e:
        e.log_metric('LB', 0.95)
        run_id_new = e.mlflow_run_id

    with open(os.path.join(tmpdir_name, 'metrics.json')) as f:
        metrics = json.load(f)

        # appended to the existing result
        assert 'LB' in metrics
        assert 'CV' in metrics

    assert run_id_old == run_id_new

    import mlflow
    client = mlflow.tracking.MlflowClient()
    old_run = client.get_run(run_id_old)
    assert old_run.info.lifecycle_stage == 'active'


def test_experiment_duplicated_rename_mlflow(tmpdir_name):
    with Experiment(tmpdir_name, with_mlflow=True) as e:
        e.log_metric('CV', 0.97)
        run_id_old = e.mlflow_run_id

    with Experiment(tmpdir_name, with_mlflow=True, if_exists='rename') as e:
        e.log_metric('LB', 0.95)
        run_id_new = e.mlflow_run_id

    assert run_id_old != run_id_new


def test_experiment_continue(tmpdir_name):
    with Experiment(tmpdir_name, with_mlflow=True) as e:
        e.log_metric('CV', 0.97)

    # appending to exising local & mlflow result
    with Experiment.continue_from(tmpdir_name, with_mlflow=True) as e:
        e.log_metric('LB', 0.95)

        metric_file = os.path.join(tmpdir_name, 'metrics.json')

        import mlflow

        client = mlflow.tracking.MlflowClient()
        data = client.get_run(mlflow.active_run().info.run_id).data
        assert data.metrics['CV'] == 0.97
        assert data.metrics['LB'] == 0.95

    with open(metric_file, 'r') as f:
        obj = json.load(f)
        assert obj['CV'] == 0.97
        assert obj['LB'] == 0.95

    with Experiment(tmpdir_name, with_mlflow=True, if_exists='append') as e:
        e.log_metric('X', 1.1)

        import mlflow

        client = mlflow.tracking.MlflowClient()
        data = client.get_run(mlflow.active_run().info.run_id).data
        assert data.metrics['CV'] == 0.97
        assert data.metrics['LB'] == 0.95
        assert data.metrics['X'] == 1.1

    # stop logging to mlflow, still continue logging on local dir
    with Experiment.continue_from(tmpdir_name, with_mlflow=False) as e:
        e.log_metric('Y', 1.1)
        import mlflow
        assert mlflow.active_run() is None

    with open(metric_file, 'r') as f:
        obj = json.load(f)
        assert 'Y' in obj

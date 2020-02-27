import json
import os
import pytest
import sys

import pandas as pd
import numpy as np

from nyaggle.experiment import Experiment
from nyaggle.testing import get_temp_directory


def test_log_params():
    with get_temp_directory() as logging_dir:
        with Experiment(logging_dir) as e:
            e.log_param('x', 1)
            e.log_param('x', 2)
            e.log_params({
                'y': 'ABC',
                'z': None,
            })

        with open(os.path.join(logging_dir, 'params.json'), 'r') as f:
            params = json.load(f)

            expected = {
                'x': 2,      # if the key is duplicated, the latter one is stored
                'y': 'ABC',
                'z': 'None'  # all non-numerical values are casted to string before logging
            }
            assert params == expected


def test_log_params_empty():
    with get_temp_directory() as logging_dir:
        with Experiment(logging_dir):
            pass

        with open(os.path.join(logging_dir, 'params.json'), 'r') as f:
            params = json.load(f)
            assert params == {}


def test_log_metrics():
    with get_temp_directory() as logging_dir:
        with Experiment(logging_dir) as e:
            e.log_metric('x', 1)
            e.log_metric('x', 2)
            e.log_metrics({
                'y': 3,
                'z': 4,
            })

        with open(os.path.join(logging_dir, 'metrics.json'), 'r') as f:
            params = json.load(f)

            expected = {
                'x': 2,
                'y': 3,
                'z': 4,
            }
            assert params == expected


def test_log_metrics_empty():
    with get_temp_directory() as logging_dir:
        with Experiment(logging_dir):
            pass

        with open(os.path.join(logging_dir, 'metrics.json'), 'r') as f:
            params = json.load(f)
            assert params == {}


def test_error_while_experiment():
    with get_temp_directory() as logging_dir:
        try:
            with Experiment(logging_dir) as e:
                e.log_metric('x', 0.5)
                e.log_param('foo', 'bar')
                e.log_numpy('np', np.zeros(100))
                e.log_dataframe('df', pd.DataFrame({'a': [1, 2, 3]}))

                raise KeyboardInterrupt()
        except KeyboardInterrupt:
            pass

        # all logs are saved even if error raised inside experiment
        with open(os.path.join(logging_dir, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
            assert metrics == {'x': 0.5}

        with open(os.path.join(logging_dir, 'params.json'), 'r') as f:
            params = json.load(f)
            assert params == {'foo': 'bar'}

        assert os.path.exists(os.path.join(logging_dir, 'np.npy'))
        assert os.path.exists(os.path.join(logging_dir, 'df.f'))


def test_experiment_duplicated_error():
    with get_temp_directory() as logging_dir:
        with Experiment(logging_dir) as e:
            e.log_metric('CV', 0.97)

        with pytest.raises(ValueError):
            with Experiment(logging_dir):
                pass

        with pytest.raises(ValueError):
            with Experiment(logging_dir, if_exists='error'):
                pass


def test_experiment_duplicated_replace():
    with get_temp_directory() as logging_dir:
        with Experiment(logging_dir) as e:
            e.log_metric('CV', 0.97)

        with Experiment(logging_dir, if_exists='replace') as e:
            e.log_metric('LB', 0.95)

        with open(os.path.join(logging_dir, 'metrics.json')) as f:
            metrics = json.load(f)

            # replaced by the new result
            assert 'LB' in metrics
            assert 'CV' not in metrics


def test_experiment_duplicated_append():
    with get_temp_directory() as logging_dir:
        with Experiment(logging_dir) as e:
            e.log_metric('CV', 0.97)

        with Experiment(logging_dir, if_exists='append') as e:
            e.log_metric('LB', 0.95)

        with open(os.path.join(logging_dir, 'metrics.json')) as f:
            metrics = json.load(f)

            # appended to the existing result
            assert 'LB' in metrics
            assert 'CV' in metrics


def test_experiment_duplicated_rename():
    with get_temp_directory() as logging_dir:
        with Experiment(logging_dir) as e:
            e.log_metric('CV', 0.97)

        with Experiment(logging_dir, if_exists='rename') as e:
            e.log_metric('LB', 0.95)

        with open(os.path.join(logging_dir, 'metrics.json')) as f:
            metrics = json.load(f)
            assert 'LB' not in metrics
            assert 'CV' in metrics

        with open(os.path.join(logging_dir + '_1', 'metrics.json')) as f:
            metrics = json.load(f)
            assert 'LB' in metrics
            assert 'CV' not in metrics


def test_experiment_duplicated_replace_mlflow():
    import mlflow

    with get_temp_directory() as logging_dir:
        with Experiment(logging_dir, with_mlflow=True) as e:
            e.log_metric('CV', 0.97)
            run_id_old = e.mlflow_run_id

        with Experiment(logging_dir, with_mlflow=True, if_exists='replace') as e:
            e.log_metric('LB', 0.95)
            run_id_new = e.mlflow_run_id

        assert run_id_old != run_id_new

        client = mlflow.tracking.MlflowClient()
        old_run = client.get_run(run_id_old)
        new_run = client.get_run(run_id_new)
        assert old_run.info.lifecycle_stage == 'deleted'
        assert new_run.info.lifecycle_stage == 'active'


def test_experiment_duplicated_append_mlflow():
    with get_temp_directory() as logging_dir:
        with Experiment(logging_dir, with_mlflow=True) as e:
            e.log_metric('CV', 0.97)
            run_id_old = e.mlflow_run_id

        with Experiment(logging_dir, with_mlflow=True, if_exists='append') as e:
            e.log_metric('LB', 0.95)
            run_id_new = e.mlflow_run_id

        with open(os.path.join(logging_dir, 'metrics.json')) as f:
            metrics = json.load(f)

            # appended to the existing result
            assert 'LB' in metrics
            assert 'CV' in metrics

        assert run_id_old == run_id_new

        import mlflow
        client = mlflow.tracking.MlflowClient()
        old_run = client.get_run(run_id_old)
        assert old_run.info.lifecycle_stage == 'active'


def test_experiment_duplicated_rename_mlflow():
    with get_temp_directory() as logging_dir:
        with Experiment(logging_dir, with_mlflow=True) as e:
            e.log_metric('CV', 0.97)
            run_id_old = e.mlflow_run_id

        with Experiment(logging_dir, with_mlflow=True, if_exists='rename') as e:
            e.log_metric('LB', 0.95)
            run_id_new = e.mlflow_run_id

        assert run_id_old != run_id_new


def test_experiment_continue():
    with get_temp_directory() as logging_dir:
        with Experiment(logging_dir, with_mlflow=True) as e:
            e.log_metric('CV', 0.97)

        # appending to exising local & mlflow result
        with Experiment.continue_from(logging_dir, with_mlflow=True) as e:
            e.log_metric('LB', 0.95)

            metric_file = os.path.join(logging_dir, 'metrics.json')

            import mlflow

            client = mlflow.tracking.MlflowClient()
            data = client.get_run(mlflow.active_run().info.run_id).data
            assert data.metrics['CV'] == 0.97
            assert data.metrics['LB'] == 0.95

        with open(metric_file, 'r') as f:
            obj = json.load(f)
            assert obj['CV'] == 0.97
            assert obj['LB'] == 0.95

        with Experiment(logging_dir, with_mlflow=True, if_exists='append') as e:
            e.log_metric('X', 1.1)

            import mlflow

            client = mlflow.tracking.MlflowClient()
            data = client.get_run(mlflow.active_run().info.run_id).data
            assert data.metrics['CV'] == 0.97
            assert data.metrics['LB'] == 0.95
            assert data.metrics['X'] == 1.1

        # stop logging to mlflow, still continue logging on local dir
        with Experiment.continue_from(logging_dir, with_mlflow=False) as e:
            e.log_metric('Y', 1.1)
            import mlflow
            assert mlflow.active_run() is None

        with open(metric_file, 'r') as f:
            obj = json.load(f)
            assert 'Y' in obj


def test_redirect_stdout():
    with get_temp_directory() as tmpdir:
        with Experiment(tmpdir, capture_stdout=True) as e:
            e.log('foo')
            print('bar')
            print('buzz', file=sys.stderr)

        with open(os.path.join(tmpdir, 'log.txt'), 'r') as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]

            assert 'foo' in lines
            assert 'bar' in lines
            assert 'buzz' not in lines  # stderr is not captured

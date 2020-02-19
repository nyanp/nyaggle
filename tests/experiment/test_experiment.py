import json
import os

from nyaggle.experiment import Experiment
from nyaggle.testing import get_temp_directory


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

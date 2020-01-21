import os
from nyaggle.experiment import Experiment
from nyaggle.testing import get_temp_directory


def test_experiment_continue():
    with get_temp_directory() as logging_dir:
        with Experiment(logging_dir, with_mlflow=True, mlflow_run_name='bar') as e:
            e.log_metric('CV', 0.97)

        # appending to exising local & mlflow result
        with Experiment.continue_from(logging_dir) as e:
            e.log_metric('LB', 0.95)

            metric_file = os.path.join(logging_dir, 'metrics.txt')

            with open(metric_file, 'r') as f:
                lines = [line.split(',') for line in f.readlines()]

                assert lines[0][0] == 'CV'
                assert lines[1][0] == 'LB'

            import mlflow

            client = mlflow.tracking.MlflowClient()
            data = client.get_run(mlflow.active_run().info.run_id).data
            assert data.metrics['CV'] == 0.97
            assert data.metrics['LB'] == 0.95

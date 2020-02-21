Advanced usage
==============================

Using low-level experiment API
-------------------------------

While nyaggle provides ``run_experiment`` as a high-level API,
``Experiment`` class can be used as a low-level API that provides primitive functionality for logging experiments.

It is useful when you want to track something other than CV, or need to implement your own CV logic.


.. code-block:: python

  from nyaggle.experiment import Experiment


  with Experiment(logging_directory='./output/') as exp:
      # log key-value pair as a parameter
      exp.log_param('lr', 0.01)
      exp.log_param('optimizer', 'adam')

      # log text
      exp.log('blah blah blah')

      # log metric
      exp.log_metric('CV', 0.85)

      # log numpy ndarray
      exp.log_numpy('predicted', predicted)

      # log pandas dataframe
      exp.log_dataframe('submission', sub, file_format='csv')

      # log any file
      exp.log_artifact('path-to-your-file')


  # you can continue logging from existing result
  with Experiment.continue_from('./output') as exp:
      ...


If you are familiar with mlflow tracking, you may notice that these APIs are similar to mlflow.
``Experiment`` can be treated as a thin wrapper if you pass ``mlflow=True`` to the constructor.


.. code-block:: python

  from nyaggle.experiment import Experiment

  with Experiment(logging_directory='./output/', with_mlflow=True) as exp:
      # logging as you want, and you can see the result in mlflow ui
      ...



Log extra parameters to run_experiment
---------------------------------------

By using ``inherit_experiment`` parameter, you can mix any additional logging with the results ``run_experiment`` will create.
In the following example, nyaggle records the result of ``run_experiment`` under the same experiment as
the parameter and metrics written outside of the function.

.. code-block:: python

  from nyaggle.experiment import Experiment, run_experiment

  with Experiment(logging_directory='./output/') as exp:

      exp.log_param('my extra param', 'bar')

      run_experiment(..., inherit_experiment=exp)

      exp.log_metrics('my extra metrics', 0.999)


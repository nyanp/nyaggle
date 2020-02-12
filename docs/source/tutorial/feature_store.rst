nyaggle.feature_store
==================================

Concept
-------------------------------

Feature engineering is one of the most important parts of Kaggle.
If you do a lot of feature engineering, it is time-consuming to calculate
features each time you build a model.

Many skilled Kagglers save their features to local disk as binary (npy, pickle or feather)
to manage their features [1]_ [2]_ [3]_ [4]_.

``feature_store`` is simple helper APIs to support feature management with feather format.


.. code-block:: python

  import pandas as pd
  import nyaggle.feature_store as fs

  def make_feature_1(df: pd.DataFrame) -> pd.DataFrame:
      return ...

  # feature 1
  feature_1 = make_feature_1(df)

  # name can be str or int
  fs.save_feature(feature_1, "my_feature_1")


``save_feature`` stores dataframe as a feather format under the feature directory (``./features`` by default).
If you want to load the feature, just call ``load_feature`` by name.

.. code-block:: python

  # feature 1
  feature_1_restored = fs.load_feature("my_feature_1")


To merge all features into the main dataframe, call ``load_features`` with the main dataframe you want to merge with.


.. code-block:: python

  train = pd.read_csv('train.csv')
  test = pd.read_csv('test.csv')
  base_df = pd.concat([train, test])

  df_with_features = fs.load_features(base_df, ["my_feature_1", "magic_1", "leaky_1"])


If you don't like separating your feature engineering code into the independent module,
``cached_feature`` decorator provides cache functionality. It automatically saves returning value as
feather, and loads it in the second call.

.. code-block:: python

  import pandas as pd
  import nyaggle.feature_store as fs

  @fs.cached_feature("my_feature_1")
  def make_feature_1(df: pd.DataFrame) -> pd.DataFrame:
      ...
      return result

  # saves automatically to features/my_feature_1.f
  feature_1 = make_feature_1(df)

  # loads from saved binary instead of calling make_feature_1
  feature_1 = make_feature_1(df)


.. note::
  The function decorated by ``cached_feature`` must return pandas DataFrame.


Use with ``run_experiment``
-------------------------------

If you pass ``feature_list`` and ``feature_directory`` parameters to ``run_experiment`` API,
nyaggle will combine specified features to the given dataframe before performing cross-validation.

List of features is logged as parameters (and of course can be seen in mlflow ui),
that makes your experiment cycle much simpler.

.. code-block:: python

  import pandas as pd
  import nyaggle.feature_store as fs
  from nyaggle.experiment import run_experiment

  run_experiment(params,
                 X_train,
                 y,
                 X_test,
                 feature_list=["my_feature_1", "magic_1", "leaky_1"],
                 feature_directory="../my_features")




Reference
-------------------------------


.. [1] https://www.kaggle.com/c/avito-demand-prediction/discussion/59881
.. [2] https://github.com/flowlight0/talkingdata-adtracking-fraud-detection
.. [3] https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/55581
.. [4] https://amalog.hateblo.jp/entry/kaggle-feature-management
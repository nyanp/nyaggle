Installation
===================================

You can install nyaggle via pip:


.. code-block:: bash

    pip install nyaggle   # Install core parts of nyaggle


nyaggle does not install the following packages by default:

- catboost
- lightgbm
- xgboost
- mlflow
- pytorch


Modules which depends on these packages won't work until you also install them.
For example, ``run_experiment`` with ``algorithm_type='xgb'``, ``'lgbm'`` and ``'cat'`` options won't work
until you also install xgboost, lightgbm and catboost respectively.

If you want to install everything required in nyaggle, This command can be used:

.. code-block:: bash

    pip install nyaggle[all]  # Install everything


If you use :code:`lang=ja` option in :code:`BertSentenceVecorizer`,
you also need to intall MeCab and mecab-python3 package to your environment.

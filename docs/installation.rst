Installation
===================================

You can install nyaggle via pip:


.. code-block:: bash

    pip install nyaggle


nyaggle does not install the following packages by pip:

- catboost
- lightgbm
- xgboost
- mlflow
- pytorch

You need to install these packages if you want to use them through nyaggle API.
For example, you need to install xgboost before calling ``run_experiment`` with ``algorithm_type='xgb'``.

To use :code:`nyaggle.nlp.BertSentenceVectorizer`, you first need to install PyTorch.
Please refer to `PyTorch installation page <https://pytorch.org/get-started/locally/#start-locally>`_
to install Pytorch to your environment.

If you use :code:`lang=ja` option in :code:`BertSentenceVecorizer`,
you also need to intall MeCab and mecab-python3 package to your environment.

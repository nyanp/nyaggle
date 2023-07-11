parameters = [
    {
        "name": "ieee-2019-17th",
        "url": "https://nbviewer.jupyter.org/github/tmheo/IEEE-Fraud-Detection-17th-Place-Solution/blob/master/notebook/IEEE-17th-Place-Solution-CatBoost-Ensemble.ipynb",  # noQA
        "competition": "ieee-fraud-detection",
        "rank": 17,
        "metric": "auc",
        "parameters": {
            'learning_rate': 0.07,
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'metric_period': 500,
            'od_wait': 500,
            'depth': 8,
        }
    },
    {
        "name": "elo-2018-11th",
        "url": "https://github.com/kangzhang0709/2019-kaggle-elo-top-11-solution",
        "competition": "elo-merchant-category-recommendation",
        "rank": 11,
        "metric": "rmse",
        "parameters": {
            'learning_rate': 0.01,
            'max_depth': 8,
            'bagging_temperature': 0.8,
            'l2_leaf_reg': 45,
            'od_type': 'Iter'
        }
    },
    {
        "name": "plasticc-2018-3rd",
        "url": "https://github.com/takashioya/plasticc/blob/master/scripts/train.py",
        "competition": "PLAsTiCC-2018",
        "rank": 3,
        "metric": "multi-class log-loss",
        "parameters": {
            'learning_rate': 0.1,
            'depth': 3,
            'loss_function': 'MultiClass',
            'colsample_bylevel': 0.7,
        }
    },
]

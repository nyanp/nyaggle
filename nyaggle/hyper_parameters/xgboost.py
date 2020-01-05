import nyaggle.hyper_parameters.util as util


_parameters = [
    {
        "name": "ieee-2019-1st",
        "url": "https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600",
        "competition": "ieee-fraud-detection",
        "rank": 1,
        "metric": "auc",
        "parameters": {
            "max_depth": 12,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.4,
            "missing": -1,
            "eval_metric": "auc",
            "tree_method": "hist"
        }
    },
    {
        "name": "womens-ml-competition-2019-1st",
        "url": "https://github.com/salmatfq/KaggleMarchMadnessFirstPlace/blob/master/win_ncaa_men.R",
        "competition": "womens-machine-learning-competition-2019",
        "rank": 1,
        "metric": "log-loss",
        "parameters": {
            "eval_metric": "mae",
            "booster": "gbtree",
            "eta": 0.02,
            "subsample": 0.35,
            "colsample_bytree": 0.7,
            "num_parallel_tree": 10,
            "min_child_weight": 40,
            "gamma": 10,
            "max_depth": 3
        }
    },

    # 2018, Santander Value Prediction Challenge
    {
        "name": "santander-2018-5th",
        "url": "https://github.com/vlarine/kaggle/blob/master/santander-value-prediction-challenge/santander.py",
        "competition": "santander-value-prediction-challenge",
        "rank": 5,
        "metric": "rmsle",
        "parameters": {
            'colsample_bytree': 0.055,
            'colsample_bylevel': 0.4,
            'gamma': 1.5,
            'learning_rate': 0.01,
            'max_depth': 5,
            'objective': 'reg:linear',
            'booster': 'gbtree',
            'min_child_weight': 10,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'eval_metric': 'rmse',
            'subsample': 0.7,
        }
    },

    # 2018, Elo Merchant Category Recommendation
    {
        "name": "elo-2018-11th",
        "url": "https://github.com/kangzhang0709/2019-kaggle-elo-top-11-solution/blob/master/Models/model_xgb.ipynb",
        "competition": "elo-merchant-category-recommendation",
        "rank": 11,
        "metric": "rmse",
        "parameters": {
            'objective': 'reg:linear',
            'booster': 'gbtree',
            'learning_rate': 0.01,
            'max_depth': 10,
            'gamma': 1.45,
            'alpha': 0.1,
            'lambda': 0.3,
            'subsample': 0.9,
            'colsample_bytree': 0.054,
            'colsample_bylevel': 0.50
        }
    },

    # 2018, DonorsChoose.org Application Screening
    {
        "name": "donorschoose-2018-1st",
        "url": "https://www.kaggle.com/shadowwarrior/1st-place-solution/notebook",
        "competition": "donorschoose-application-screening",
        "rank": 1,
        "metric": "auc",
        "parameters": {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': 0.01,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.4,
            'min_child_weight': 10,
            'gamma': 2
        }
    },

    # 2018, Recruit Restaurant Visitor Forecasting

    # 2017, Instacart Market Basket Analysis
    {
        "name": "instacart-2017-2nd",
        "url": "https://github.com/KazukiOnodera/Instacart/blob/master/py_model/002_xgb_holdout_item_812_1.py",
        "competition": "instacart-market-basket-analysis",
        "rank": 2,
        "metric": "",
        "parameters": {
            'max_depth': 10,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.75,
            'eval_metric': 'logloss',
            'objective': 'binary:logistic',
            'tree_method': 'hist'
         }
    },

    # 2017, Two Sigma Connect; Rental Listing Inquiries
    {
        "name": "two-sigma-2017-1st",
        "url": "https://github.com/plantsgo/Rental-Listing-Inquiries/blob/master/xgb.py",
        "competition": "two-sigma-connect-rental-listing-inquiries",
        "rank": 1,
        "metric": "multi-class log-loss",
        "parameters": {
            'booster': 'gbtree',
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'gamma': 1,
            'min_child_weight': 1.5,
            'max_depth': 5,
            'lambda': 10,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.7,
            'eta': 0.03,
            'tree_method': 'exact'
        }
    },

    # 2016, Santander Product Recommendation
    {
        "name": "santander-2016-2nd",
        "url": "https://github.com/ttvand/Santander-Product-Recommendation/blob/master/First%20level%20learners/xgboost.R",
        "competition": "santander-product-recommendation",
        "rank": 2,
        "metric": "map7",
        "parameters": {
            "etaC": 10,
            "subsample": 1,
            "colsample_bytree": 0.5,
            "max_depth": 8,
            "min_child_weight": 0,
            "gamma": 0.1
        }
    },

    # 2016, TalkingData Mobile User Demographics
    {
        "name": "talkingdata-2016-3rd-1",
        "url": "https://github.com/chechir/talking_data/blob/master/danijel/xgb/xgb_cv5_train_events.R",
        "competition": "talkingdata-mobile-user-demographics",
        "rank": 3,
        "metric": "multi-class log-loss",
        "parameters": {
            "booster": 'gbtree',
            "objective": 'reg:logistic',
            "eval_metric": 'logloss',
            "learning_rate": 0.025,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.5,
            "colsample_bylevel": 0.5
        }
    },
    {
        "name": "talkingdata-2016-3rd-2",
        "url": "https://github.com/chechir/talking_data/blob/master/danijel/xgb/xgb_cv5_train_noevents.R",
        "competition": "talkingdata-mobile-user-demographics",
        "rank": 3,
        "metric": "multi-class log-loss",
        "parameters": {
            "booster": 'gbtree',
            "objective": 'reg:logistic',
            "eval_metric": 'logloss',
            "learning_rate": 0.05,
            "max_depth": 2,
            "colsample_bytree": 0.8,
            "colsample_bylevel": 0.8
        }
    },

    # 2016, Allstate Claims Severity
    {
        "name": "allstate-2016-3rd",
        "url": "https://www.kaggle.com/c/allstate-claims-severity/discussion/26447#150319",
        "competition": "allstate-claims-severity",
        "rank": 3,
        "metric": "mae",
        "parameters": {
            'colsample_bytree': 0.4,
            'subsample': 0.975,
            'learning_rate': 0.015,
            'gamma': 1.5,
            'lambda': 2,
            'alpha': 2,
            'max_depth': 25,
            'num_parallel_tree': 1,
            'min_child_weight': 50,
            'eval_metric': 'mae',
            'max_delta_step': 0,
        }
    },

    # 2016, Bosch Production Line Performance
    {
        "name": "bosch-2016-1st",
        "url": "https://www.kaggle.com/c/bosch-production-line-performance/discussion/25434#144628",
        "competition": "bosch-production-line-performance",
        "rank": 1,
        "metric": "mcc",
        "parameters": {
            "eval_metric": "auc",
            "alpha": 0,
            "booster": "gbtree",
            "colsample_bytree": 0.6,
            "minchildweight": 5,
            "subsample": 0.9,
            "eta": 0.03,
            "objective": "binary:logistic",
            "max_depth": 14,
            "lambda": 4
        }
    },
]


def list_param_names():
    return util.list_param_names(_parameters)


def get_hyperparam_byname(name: str, with_metadata: bool = False):
    return util.get_hyperparam_byname(_parameters, name, with_metadata)


def get_all_hyperparams():
    return _parameters

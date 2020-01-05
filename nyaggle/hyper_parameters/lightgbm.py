import nyaggle.hyper_parameters.util as util

# seed, verbosity, num_boosting_rounds, num_class, num_thread are removed from original code
# (too specific to the competition or environment)

_parameters = [
    # 2019, ASHRAE - Great Energy Predictor III

    # 2019, IEEE-CIS Fraud Detection
    {
        "name": "ieee-2019-10th",
        "url": "https://github.com/jxzly/Kaggle-IEEE-CIS-Fraud-Detection-2019/blob/master/main.py",
        "competition": "ieee-fraud-detection",
        "rank": 10,
        "metric": "auc",
        "parameters": {
            'num_leaves': 333,
            'min_child_weight': 0.03454472573214212,
            'feature_fraction': 0.3797454081646243,
            'bagging_fraction': 0.4181193142567742,
            'min_data_in_leaf': 106,
            'objective': 'binary',
            'max_depth': -1,
            'learning_rate': 0.006883242363721497,
            "boosting_type": "gbdt",
            "metric": 'auc',
            'reg_alpha': 0.3899927210061127,
            'reg_lambda': 0.6485237330340494
         }
    },
    {
        "name": "ieee-2019-17th",
        "url": "https://nbviewer.jupyter.org/github/tmheo/IEEE-Fraud-Detection-17th-Place-Solution/blob/master/notebook/IEEE-17th-Place-Solution-LightGBM.ipynb",
        "competition": "ieee-fraud-detection",
        "rank": 17,
        "metric": "auc",
        "parameters": {
            'objective': 'binary',
            'metric': 'None',
            'learning_rate': 0.01,
            'num_leaves': 256,
            'max_bin': 255,
            'max_depth': -1,
            'bagging_freq': 5,
            'bagging_fraction': 0.7,
            'feature_fraction': 0.7,
            'first_metric_only': True,
        }
    },

    # 2019, Predicting Molecular Properties

    # 2019, Instant Gratification

    # 2019, LANL Earthquake Prediction
    {
        "name": "lanl-2019-1st",
        "url": "https://www.kaggle.com/ilu000/1-private-lb-kernel-lanl-lgbm/",
        "competition": "LANL-Earthquake-Prediction",
        "rank": 1,
        "metric": "mae",
        "parameters": {
            'num_leaves': 4,
            'min_data_in_leaf': 5,
            'objective': 'fair',
            'max_depth': -1,
            'learning_rate': 0.02,
            "boosting": "gbdt",
            'boost_from_average': True,
            "feature_fraction": 0.9,
            "bagging_freq": 1,
            "bagging_fraction": 0.5,
            'max_bin': 500,
            'reg_alpha': 0,
            'reg_lambda': 0
        }
    },
    {
        "name": "lanl-2019-10th",
        "url": "https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94466",
        "competition": "LANL-Earthquake-Prediction",
        "rank": 10,
        "metric": "mae",
        "parameters": {
            "objective": "gamma",
            "max_depth": -1,
            "feature_fraction": 0.025,
            "bagging_fraction": 0.250,
            "bagging_freq": 1,
            "num_leaves": 7,
            "min_data_in_bin": 2,
            "max_bin": 25,
            "min_data_in_leaf": 4,
            "lambda_l1": 1.1,
            "lambda_l2": 0.1,
            "learning_rate": 0.01
        }
    },

    # 2019, Santander Customer Transaction Prediction
    {
        "name": "santander-2019-2nd",
        "url": "https://github.com/KazukiOnodera/santander-customer-transaction-prediction/blob/master/py/990_2nd_place_solution_golf.py",
        "competition": "santander-customer-transaction-prediction",
        "rank": 2,
        "metric": "auc",
        "parameters": {
            'bagging_freq': 5,
            'bagging_fraction': 1.0,
            'boost_from_average': 'false',
            'boost': 'gbdt',
            'feature_fraction': 1.0,
            'learning_rate': 0.005,
            'max_depth': -1,
            'min_data_in_leaf': 30,
            'min_sum_hessian_in_leaf': 10.0,
            'num_leaves': 64,
            'tree_learner': 'serial',
            'metric': 'binary_logloss',
            'objective': 'binary',
        }
    },
    {
        "name": "santander-2019-5th",
        "url": "https://github.com/tnmichael309/Kaggle-Santander-Customer-Transaction-Prediction-5th-Place-Partial-Solution/blob/master/notebooks/LGB%20Model.ipynb",
        "competition": "santander-customer-transaction-prediction",
        "rank": 5,
        "metric": "auc",
        "parameters": {
            'num_leaves': 8,
            'min_data_in_leaf': 42,
            'objective': 'binary',
            'max_depth': 16,
            'learning_rate': 0.03,
            'boosting': 'gbdt',
            'bagging_freq': 5,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8201,
            'reg_alpha': 1.7289,
            'reg_lambda': 4.984,
            'metric': 'auc',
            'subsample': 0.81,
            'min_gain_to_split': 0.01,
            'min_child_weight': 19.428
        }
    },

    # 2019, PetFinder.my Adoption Prediction
    {
        "name": "petfinder-2019-2nd",
        "url": "https://github.com/okotaku/pet_finder/blob/master/code/stack-480-speedup.py",
        "competition": "petfinder-adoption-prediction",
        "rank": 2,
        "metric": "qwk",
        "parameters": {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'num_leaves': 63,
            'subsample': 0.9,
            'subsample_freq': 1,
            'colsample_bytree': 0.6,
            'max_depth': 9,
            'max_bin': 127,
            'reg_alpha': 0.11,
            'reg_lambda': 0.01,
            'min_child_weight': 0.2,
            'min_child_samples': 20,
            'min_gain_to_split': 0.02,
            'min_data_in_bin': 3,
            'bin_construct_sample_cnt': 5000,
            'cat_l2': 10
        }
    },

    # 2018, VSB Power Line Fault Detection
    {
        "name": "vsb-2018-1st",
        "url": "https://www.kaggle.com/mark4h/vsb-1st-place-solution",
        "competition": "vsb-power-line-fault-detection",
        "rank": 1,
        "metric": "mcc",
        "parameters": {
            'objective': 'binary',
            'boosting': 'gbdt',
            'metric': 'binary_logloss',
            'learning_rate': 0.01,
            'num_leaves': 80,
            'feature_fraction': 0.8,
            'bagging_freq': 1,
            'bagging_fraction': 0.8
        }
    },

    # 2018, Microsoft Malware Prediction
    {
        "name": "microsoft-2018-2nd",
        "url": "https://github.com/imor-de/microsoft_malware_prediction_kaggle_2nd/blob/master/code/7_Submission_M2.ipynb",
        "competition": "microsoft-malware-prediction",
        "rank": 2,
        "metric": "auc",
        "parameters": {
            'boosting_type': 'gbdt',
            'colsample_bytree': 0.6027132059774907,
            'learning_rate': 0.010899921631042043,
            'min_child_samples': 145,
            'num_leaves': 156,
            'reg_alpha': 0.45996805852518485,
            'reg_lambda': 0.7336912016500579,
            'subsample_for_bin': 440000,
            'subsample': 0.5512957111882841
        }
    },
    {
        "name": "microsoft-2018-4th",
        "url": "https://github.com/Johnnyd113/Microsoft-Malware-Prediction/blob/master/train.py",
        "competition": "microsoft-malware-prediction",
        "rank": 4,
        "metric": "auc",
        "parameters": {
            'num_leaves': 128,
            'min_data_in_leaf': 42,
            'objective': 'binary',
            'metric': 'auc',
            'max_depth': -1,
            'learning_rate': 0.05,
            "boosting": "gbdt",
            "feature_fraction": 0.8,
            "bagging_freq": 5,
            "bagging_fraction": 0.8,
            "lambda_l1": 0.15,
            "lambda_l2": 0.15
        }
    },

    # 2018, Elo Merchant Category Recommendation
    {
        "name": "elo-2018-11th",
        "url": "https://github.com/kangzhang0709/2019-kaggle-elo-top-11-solution/blob/master/Models/model_lgbm.ipynb",
        "competition": "elo-merchant-category-recommendation",
        "rank": 11,
        "metric": "rmse",
        "parameters": {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'subsample': 0.78,
            'max_depth': 8,
            'num_leaves': 63,
            'min_child_weight': 41.9612,
            'reg_alpha': 9.677,
            'colsample_bytree': 0.566,
            'min_split_gain': 8.820,
            'reg_lambda': 9.253,
            'min_data_in_leaf': 21,
        }
    },

    # 2018, Google Analytics Customer Revenue Prediction
    {
        "name": "ga-2018-1st",
        "url": "https://www.kaggle.com/kostoglot/winning-solution",
        "competition": "ga-customer-revenue-prediction",
        "rank": 1,
        "metric": "rmse",
        "parameters": {
            "objective": "regression",
            "max_bin": 256,
            "learning_rate": 0.01,
            "num_leaves": 9,
            "bagging_fraction": 0.9,
            "feature_fraction": 0.8,
            "min_data": 1,
            "bagging_freq": 1,
            "metric": "rmse"
        }
    },

    # 2018, PLAsTiCC Astronomical Classification
    {
        "name": "plasticc-2018-1st",
        "url": "https://github.com/kboone/avocado/blob/master/avocado/classifier.py",
        "competition": "PLAsTiCC-2018",
        "rank": 1,
        "metric": "multi-class log-loss",
        "parameters": {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "colsample_bytree": 0.5,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "min_split_gain": 10.0,
            "min_child_weight": 2000.0,
            "max_depth": 7,
            "num_leaves": 50,
        }
    },
    {
        "name": "plasticc-2018-3rd",
        "url": "https://github.com/nyanp/kaggle-PLASTiCC/blob/master/model/experiment65.py",
        "competition": "PLAsTiCC-2018",
        "rank": 3,
        "metric": "multi-class log-loss",
        "parameters": {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'subsample': .9,
            'colsample_bytree': .9,
            'reg_alpha': 0,
            'reg_lambda': 3,
            'min_split_gain': 0,
            'min_child_weight': 10,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_data_in_leaf': 1,
            'max_bin': 128,
            'bagging_fraction': 0.66
        }
    },

    # 2018, Home Credit Default Risk
    {
        "name": "home-credit-2018-2nd",
        "url": "https://github.com/KazukiOnodera/Home-Credit-Default-Risk/blob/master/py/935_predict_908-2.py",
        "competition": "home-credit-default-risk",
        "rank": 2,
        "metric": "auc",
        "parameters": {
             'objective': 'binary',
             'metric': 'auc',
             'learning_rate': 0.01,
             'max_depth': 6,
             'num_leaves': 63,
             'max_bin': 255,
             'min_child_weight': 10,
             'min_data_in_leaf': 150,
             'reg_lambda': 0.5,
             'reg_alpha': 0.5,
             'colsample_bytree': 0.7,
             'subsample': 0.9,
             'bagging_freq': 1,
         }
    },
    {
        "name": "home-credit-2018-7th",
        "url": "https://github.com/js-aguiar/home-credit-default-competition/blob/master/config.py",
        "competition": "home-credit-default-risk",
        "rank": 7,
        "metric": "auc",
        "parameters": {
            'boosting_type': 'goss',
            'learning_rate': 0.005134,
            'num_leaves': 54,
            'max_depth': 10,
            'subsample_for_bin': 240000,
            'reg_alpha': 0.436193,
            'reg_lambda': 0.479169,
            'colsample_bytree': 0.508716,
            'min_split_gain': 0.024766,
            'subsample': 1
        }
    },

    # 2018, Santander Value Prediction Challenge
    {
        "name": "santander-2018-6th",
        "url": "https://www.kaggle.com/joeytaj/leak-fe-ml-from-scratch-baseline/notebook",
        "competition": "santander-value-prediction-challenge",
        "rank": 6,
        "metric": "rmsle",
        "parameters": {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'gbdt',
            'is_training_metric': True,
            'max_bin': 350,
            'learning_rate': .005,
            'max_depth': -1,
            'num_leaves': 48,
            'feature_fraction': 0.1,
            'reg_alpha': 0,
            'reg_lambda': 0.2,
            'min_child_weight': 10
        }
    },

    # 2018, Avito Demand Prediction Challenge
    {
        "name": "avito-2018-4th",
        "url": "https://www.kaggle.com/c/avito-demand-prediction/discussion/59881",
        "competition": "avito-demand-prediction",
        "rank": 4,
        "metric": "rmse",
        "parameters": {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'num_leaves': 400,
            'colsample_bytree': .45
        }
    },
    {
        "name": "avito-2018-5th",
        "url": "https://github.com/darraghdog/avito-demand/blob/master/lgb/lb_1406.py",
        "competition": "avito-demand-prediction",
        "rank": 5,
        "metric": "rmse",
        "parameters": {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective' : 'regression',
            'metric' : 'rmse',
            'num_leaves' : 1000,
            'learning_rate' : 0.02,
            'feature_fraction' : 0.5,
        }
    },

    # 2018, TalkingData AdTracking Fraud Detection Challenge
    {
        "name": "talkingdata-2018-1st",
        "url": "https://github.com/flowlight0/talkingdata-adtracking-fraud-detection/blob/master/configs/lightgbm_119.json",
        "competition": "talkingdata-adtracking-fraud-detection",
        "rank": 1,
        "metric": "auc",
        "parameters": {
            "boosting_type": "gbdt",
            "objective": "binary",
            "learning_rate": 0.01,
            "num_leaves": 255,
            "max_depth": 8,
            "min_child_samples": 200,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.5,
            "min_child_weight": 0,
            "subsample_for_bin": 1000000,
            "min_split_gain": 0,
            "reg_lambda": 0,
        }
    },
    {
        "name": "porto-seguro-2017-2nd",
        "url": "https://www.kaggle.com/xiaozhouwang/2nd-place-lightgbm-solution",
        "competition": "porto-seguro-safe-driver-prediction",
        "rank": 2,
        "metric": "normalized-gini-coefficient",
        "parameters": {
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 15,
            "max_bin": 256,
            "feature_fraction": 0.6,
            "drop_rate": 0.1,
            "is_unbalance": False,
            "max_drop": 50,
            "min_child_samples": 10,
            "min_child_weight": 150,
            "min_split_gain": 0,
            "subsample": 0.9
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
          'objective': 'binary',
          'metric': 'auc',
          'boosting_type': 'dart',
          'learning_rate': 0.01,
          'max_bin': 15,
          'max_depth': 17,
          'num_leaves': 63,
          'subsample': 0.8,
          'subsample_freq': 5,
          'colsample_bytree': 0.8,
          'reg_lambda': 7
        }
    },

    # 2018, Recruit Restaurant Visitor Forecasting
    {
        "name": "recruit-2018-1st",
        "url": "https://www.kaggle.com/plantsgo/solution-public-0-471-private-0-505",
        "competition": "recruit-restaurant-visitor-forecasting",
        "rank": 1,
        "metric": "rmse",
        "parameters": {
            'num_leaves': 255,
            'objective': 'regression_l2',
            'max_depth': 9,
            'min_data_in_leaf': 50,
            'learning_rate': 0.007,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'metric': 'rmse'
        }
    },
    {
        "name": "recruit-2018-1st",
        "url": "https://github.com/MaxHalford/kaggle-recruit-restaurant/blob/master/Solution.ipynb",
        "competition": "recruit-restaurant-visitor-forecasting",
        "rank": 8,
        "metric": "rmse",
        "parameters": {
            "objective": 'regression',
            "max_depth": 5,
            "num_leaves": 24,
            "learning_rate": 0.007,
            "n_estimators": 30000,
            "min_child_samples": 80,
            "subsample": 0.8,
            "colsample_bytree": 1,
            "reg_alpha": 0,
            "reg_lambda": 0,
        }
    },
]


def list_param_names():
    return util.list_param_names(_parameters)


def get_hyperparam_byname(name: str, with_metadata: bool = False):
    return util.get_hyperparam_byname(_parameters, name, with_metadata)


def get_all_hyperparams():
    return _parameters

import pandas as pd

from sklearn.model_selection import StratifiedKFold
from nyaggle.experiment import experiment_gbdt


meta = pd.read_csv('training_set_metadata.csv')

is_extra = meta.hostgal_photoz > 0.0
meta_extra = meta[is_extra]
meta_inner = meta[~is_extra]

lgb_param_extra = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 9
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

result_extra = experiment_gbdt(lgb_param_extra,
                               meta_extra.drop('target', axis=1),
                               meta_extra['target'],
                               logging_directory='plasticc-{time}',
                               cv=skf,
                               type_of_target='multiclass')

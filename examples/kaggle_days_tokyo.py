import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from nyaggle.experiment import experiment_gbdt
from nyaggle.feature.category_encoder import TargetEncoder

lgb_params = {
    "objective": "rmse",
    "n_estimators": 2000,
    "max_depth": 10,
    "colsample_bytree": 0.8
}

X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
y_train = X_train['age']
X_train = X_train.drop('age', axis=1)

te_cols = [c for c in X_train.columns if X_train[c].dtype.name == 'object' and c not in ['user_id', 'ts']]
te = TargetEncoder(cv=GroupKFold(5), cols=te_cols, groups=X_train['user_id']).fit(X_train, y_train)


def transform(te: TargetEncoder, df: pd.DataFrame, y: pd.Series):
    df.drop('ts', axis=1, inplace=True)

    if y is not None:
        df = te.fit_transform(df, y)
        y = y.groupby(df['user_id']).min()
    else:
        df = te.transform(df)

    df = df.groupby('user_id').agg(['mean', 'min', 'max'])
    df.columns = [e[0] + '_' + e[1] for e in df.columns]
    return df.reset_index(), y


X_train, y_train = transform(te, X_train, y_train)
X_test, _ = transform(te, X_test, None)

# generated submission.csv scores 11.61445 in private LB (35th)
experiment_gbdt(logging_directory='baseline_kaggledays_tokyo',
                model_params=lgb_params,
                id_column='user_id',
                X_train=X_train,
                y=y_train,
                X_test=X_test,
                eval=mean_squared_error,
                type_of_target='continuous',
                overwrite=True)

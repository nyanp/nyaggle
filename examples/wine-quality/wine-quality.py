import pandas as pd
from sklearn.model_selection import train_test_split

from nyaggle.experiment import experiment_gbdt


csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

data = pd.read_csv(csv_url, sep=';')

X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


params = {
    'max_depth': 4,
    'n_estimators': 1000,
    'reg_alpha': 0.1
}

result = experiment_gbdt(params,
                         X_train,
                         y_train,
                         X_test,
                         './wine-quality-{time}',
                         type_of_target='continuous',
                         with_mlflow=True,
                         with_auto_hpo=True)

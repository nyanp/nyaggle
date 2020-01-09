import numpy as np
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

from nyaggle.validation import adversarial_validate
from nyaggle.testing import make_classification_df


def test_adv():
    X, y = make_classification_df(1024)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    X_train['target'] = 0
    X_test['target'] = 1

    auc, importance = adversarial_validate(X_train, X_test)

    assert importance['feature'][0] == 'target'
    assert auc >= 0.9

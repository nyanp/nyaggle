import os
import tempfile

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from nyaggle.experiment import average_results, experiment
from nyaggle.testing import make_classification_df


def test_averaging():
    X, y = make_classification_df(n_samples=1024, n_num_features=10, n_cat_features=2,
                                  class_sep=0.98, random_state=0, id_column='user_id')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    params = {
        'objective': 'binary',
        'max_depth': 8
    }

    with tempfile.TemporaryDirectory() as temp_path:
        for i in range(3):
            params['seed'] = i
            ret_single = experiment(params, X_train, y_train, X_test,
                                    os.path.join(temp_path, 'seed{}'.format(i)))

        df = average_results([
            os.path.join(temp_path, 'seed{}'.format(i)) for i in range(3)
        ], os.path.join(temp_path, 'average.csv'))

        score = roc_auc_score(y_test, df[df.columns[-1]])
        assert score >= 0.85

        assert score >= roc_auc_score(y_test, ret_single.test_prediction)

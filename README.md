# nyaggle
**nyaggle** is a utility library for Kaggle and offline competitions, 
particularly focused on feature engineering and validation. 
See [the documentation](https://nyaggle.readthedocs.io/en/latest/index.html) for details.

- Feature Engineering
    - K-Fold Target Encoding
    - BERT Sentence Vectorization
- Model Validation
    - CV with OOF
- GBDT Experiment Package
    - Generate CV score, submission.csv, OOF, importance plot at once
- Ensemble
    - Blending

## Installation
You can install nyaggle via pip:
```
$pip install nyaggle
```

## Examples

### Feature Engineering

#### Target Encoding with K-Fold
```python
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from nyaggle.feature.category_encoder import TargetEncoder


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
all = pd.concat([train, test]).copy()

cat_cols = [c for c in train.columns if train[c].dtype == np.object]
target_col = 'y'

kf = KFold(5)

# Target encoding with K-fold
te = TargetEncoder(split=kf.split(train))

# use fit/fit_transform to train data, then apply transform to test data
train.loc[:, cat_cols] = te.fit_transform(train[cat_cols], train[target_col])
test.loc[:, cat_cols] = te.transform(test[cat_cols])

# ... or just call fit_transform to concatenated data
all.loc[:, cat_cols] = te.fit_transform(all[cat_cols], all[cat_cols])
```

#### Text Vectorization using BERT
You need to install pytorch to your virtual environment to use BertSentenceVectorizer. 
MaCab and mecab-python3 are also required if you use Japanese BERT model.

```python
import pandas as pd
from nyaggle.feature.nlp import BertSentenceVectorizer


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
all = pd.concat([train, test]).copy()

text_cols = ['body']
target_col = 'y'
group_col = 'user_id'


# extract BERT-based sentence vector
bv = BertSentenceVectorizer(text_columns=text_cols)

text_vector = bv.fit_transform(train)


# BERT + SVD, with cuda
bv = BertSentenceVectorizer(text_columns=text_cols, use_cuda=True, n_components=40)

text_vector_svd = bv.fit_transform(train)

# Japanese BERT
bv = BertSentenceVectorizer(text_columns=text_cols, lang='jp')

japanese_text_vector = bv.fit_transform(train)
```

### Model Validation
`cv()` provides handy API to calculate K-fold CV, Out-of-Fold prediction and test prediction at one time.
You can pass LGBMClassifier/LGBMRegressor and any other sklearn models.

```python
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from nyaggle.validation import cross_validate

X, y = make_classification(n_samples=1024, n_features=20, class_sep=0.98, random_state=0)

models = [LGBMClassifier(n_estimators=300) for _ in range(5)]

pred_oof, pred_test, scores = cross_validate(models, X[:512, :], y[:512], X[512:, :], nfolds=5,
                                             eval=roc_auc_score)
```

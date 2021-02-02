from typing import Any, Callable, List, Optional, Union
import transformers

import numpy as np
import pandas as pd
from category_encoders.utils import convert_input
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from nyaggle.environment import requires_torch
from nyaggle.feature.base import BaseFeaturizer


class BertSentenceVectorizer(BaseFeaturizer):
    """Sentence Vectorizer using BERT pretrained model.

    Extract fixed-length feature vector from English/Japanese variable-length sentence using BERT.

    Args:
        lang:
            Type of language. If set to "jp", Japanese BERT model is used (you need to install MeCab).
        n_components:
            Number of components in SVD. If `None`, SVD is not applied.
        text_columns:
            List of processing columns. If `None`, all object columns are regarded as text column.
        pooling_strategy:
            The pooling algorithm for generating fixed length encoding vector. 'reduce_mean' and 'reduce_max' use
            average pooling and max pooling respectively to reduce vector from (num-words, emb-dim) to (emb_dim).
            'reduce_mean_max' performs 'reduce_mean' and 'reduce_max' separately and concat them.
            'cls_token' takes the first element (i.e. [CLS]).
        use_cuda:
            If `True`, inference is performed on GPU.
        tokenizer:
            The custom tokenizer used instead of default tokenizer
        model:
            The custom pretrained model used instead of default BERT model
        return_same_type:
            If True, `transform` and `fit_transform` return the same type as X.
            If False, these APIs always return a numpy array, similar to sklearn's API.
        column_format:
            Name of transformed columns (used if returning type is pd.DataFrame)
    """

    def __init__(self, lang: str = 'en', n_components: Optional[int] = None,
                 text_columns: List[str] = None, pooling_strategy: str = 'reduce_mean',
                 use_cuda: bool = False, tokenizer: transformers.PreTrainedTokenizer = None,
                 model=None, return_same_type: bool = True, column_format: str = '{col}_{idx}'):
        if tokenizer is not None:
            assert model is not None
            self.tokenizer = tokenizer
            self.model = model
        if lang == 'en':
            pretrained_model_name = 'bert-base-uncased'
            self.tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_model_name)
            self.model = transformers.BertModel.from_pretrained(pretrained_model_name)
        elif lang == 'jp':
            pretrained_model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
            self.tokenizer = transformers.BertJapaneseTokenizer.from_pretrained(pretrained_model_name)
            self.model = transformers.BertModel.from_pretrained(pretrained_model_name)
        else:
            raise ValueError('Specified language type () is invalid.'.format(lang))

        self.lang = lang
        self.n_components = n_components
        self.text_columns = text_columns
        self.pooling_strategy = pooling_strategy
        self.use_cuda = use_cuda
        self.return_same_type = return_same_type
        self.svd = {}
        self.column_format = column_format

    def _process_text(self, text: str) -> np.ndarray:
        requires_torch()
        import torch

        tokens_tensor = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)])
        if self.use_cuda:
            tokens_tensor = tokens_tensor.to('cuda')
            self.model.to('cuda')

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tokens_tensor)

        embedding = outputs.last_hidden_state.cpu().numpy()[0]
        if self.pooling_strategy == 'reduce_mean':
            return np.mean(embedding, axis=0)
        elif self.pooling_strategy == 'reduce_max':
            return np.max(embedding, axis=0)
        elif self.pooling_strategy == 'reduce_mean_max':
            return np.r_[np.max(embedding, axis=0), np.mean(embedding, axis=0)]
        elif self.pooling_strategy == 'cls_token':
            return embedding[0]
        else:
            raise ValueError("specify valid pooling_strategy: {reduce_mean, reduce_max, reduce_mean_max, cls_token}")

    def _fit_one(self, col: str, emb: np.ndarray):
        if not self.n_components or self.n_components >= emb.shape[1]:
            return emb
        self.svd[col] = TruncatedSVD(n_components=self.n_components, algorithm='arpack', random_state=0)
        return self.svd[col].fit(emb)

    def _transform_one(self, col: str, emb: np.ndarray):
        if not self.n_components or self.n_components >= emb.shape[1]:
            return emb
        return self.svd[col].transform(emb)

    def _fit_transform_one(self, col: str, emb: np.ndarray):
        if not self.n_components or self.n_components >= emb.shape[1]:
            return emb
        self.svd[col] = TruncatedSVD(n_components=self.n_components, algorithm='arpack', random_state=0)
        return self.svd[col].fit_transform(emb)

    def _process(self, X: pd.DataFrame, func: Callable[[str, np.ndarray], Any]):
        is_pandas = isinstance(X, pd.DataFrame)
        X = convert_input(X)

        tqdm.pandas()
        columns = self.text_columns or [c for c in X.columns if X[c].dtype == np.object]
        non_text_columns = [c for c in X.columns if c not in columns]

        column_names = []
        processed = []
        for c in columns:
            emb = np.vstack(X[c].progress_apply(lambda x: self._process_text(x)))
            emb = func(c, emb)
            processed.append(emb)
            column_names += [self.column_format.format(col=c, idx=i) for i in range(emb.shape[1])]

        processed_df = pd.DataFrame(np.hstack(processed), columns=column_names)

        if non_text_columns:
            X_ = X[non_text_columns].copy()
            X_ = pd.concat([X_, processed_df], axis=1)
        else:
            X_ = processed_df

        return X_ if self.return_same_type and is_pandas else X_.values

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fit SVD model on training data X.

        Args:
            X:
                Data
            y:
                Ignored
        """
        self._process(X, self._fit_one)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Perform feature extraction and dimensionality reduction using
        BERT pre-trained model and trained SVD model.

        Args:
            X:
                Data
            y:
                Ignored
        """
        return self._process(X, self._transform_one)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y=None, **fit_params):
        """
        Fit SVD model on training data X and perform feature extraction and dimensionality reduction using
        BERT pre-trained model and trained SVD model.

        Args:
            X:
                Data
            y:
                Ignored
        """
        return self._process(X, self._fit_transform_one)

import pytest

import numpy.testing as npt
import pandas as pd

from nyaggle import Language
from nyaggle.feature.nlp import BertSentenceVectorizer


_TEST_SENTENCE_EN = [
    'This is a pen.',
    'A quick brown fox',
    'Redistribution and use in source and binary forms, with or without modification.',
    'BERT is the state of the art NLP model.',
    'This is a pen.',
    'THIS IS A PEN.',
]

_TEST_SENTENCE_JP = [
    '金メダルが5枚欲しい。',
    '私は昨日から風邪をひいています。',
    'これはペンです。',
    'BERTは最新の自然言語処理モデルです。',
    '金メダルが5枚欲しい。',
    '金メダルが 5枚 欲しい。',
]


def test_bert_en():
    bert = BertSentenceVectorizer(use_cuda=False)

    X = pd.DataFrame({
        'id': [0, 1, 2, 3, 4, 5],
        'sentence': _TEST_SENTENCE_EN
    })

    ret = bert.fit_transform(X)

    assert ret.shape[0] == 6
    assert ret.shape[1] == 768
    npt.assert_almost_equal(ret[0], ret[4])
    npt.assert_almost_equal(ret[0], ret[5])


def test_bert_en_svd():
    n_components = 3
    bert = BertSentenceVectorizer(n_components=n_components, use_cuda=False)

    X = pd.DataFrame({
        'id': [0, 1, 2, 3, 4, 5],
        'sentence': _TEST_SENTENCE_EN
    })

    ret = bert.fit_transform(X)

    assert ret.shape[0] == 6
    assert ret.shape[1] == n_components
    npt.assert_almost_equal(ret[0], ret[4], decimal=3)
    npt.assert_almost_equal(ret[0], ret[5], decimal=3)


def test_bert_en_svd_multicol():
    bert = BertSentenceVectorizer(use_cuda=False)

    X = pd.DataFrame({
        'id': [0, 1, 2, 3, 4, 5],
        'sentence': _TEST_SENTENCE_EN,
        'sentence2': _TEST_SENTENCE_EN
    })

    ret = bert.fit_transform(X)

    assert ret.shape[0] == 6
    assert ret.shape[1] == 2*768
    npt.assert_almost_equal(ret[0], ret[4], decimal=3)
    npt.assert_almost_equal(ret[0], ret[5], decimal=3)


def test_bert_jp():
    bert = BertSentenceVectorizer(use_cuda=False, lang=Language.JP)

    X = pd.DataFrame({
        'id': [0, 1, 2, 3, 4, 5],
        'sentence': _TEST_SENTENCE_JP
    })

    ret = bert.fit_transform(X)

    assert ret.shape[0] == 6
    assert ret.shape[1] == 768
    npt.assert_almost_equal(ret[0], ret[4])
    npt.assert_almost_equal(ret[0], ret[5])



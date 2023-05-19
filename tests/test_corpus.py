import pytest
import pandas as pd
import json
from sktalk.corpus import Corpus


@pytest.fixture
def test_ulwa():
    return Corpus(path='data/ulwa_testdata_sktalk_format.csv')


def test_init(test_ulwa):
    assert isinstance(test_ulwa.df, pd.DataFrame)

    ulwa_json = json.loads(test_ulwa.json)
    assert ulwa_json is not None
    assert isinstance(ulwa_json, dict)

    expected_num_rows = 10
    expected_num_cols = 7
    assert test_ulwa.df.shape == (expected_num_rows, expected_num_cols)

    expected_columns = ['begin', 'end', 'translation']
    assert all(colname in test_ulwa.df.columns for colname in expected_columns)


def test_extract_speakers(test_ulwa):
    test_ulwa.extract_speakers()

    expected_speakers = ['Tang', 'Yan']
    assert all(speaker in test_ulwa.speakers for speaker in expected_speakers)

    wrong_speakers = ['Steve', 'Mary']
    assert all(speaker not in test_ulwa.speakers for speaker in wrong_speakers)

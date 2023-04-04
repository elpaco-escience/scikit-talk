"""Tests for the corpusbuilder
"""
import pytest
import pandas as pd

import sktalk.corpus as corpus

def test_corpusbuilder():
    cp = corpus.Corpus("data/vamale_testdata_sktalk_format.csv")
    assert(type(cp.content) == pd.DataFrame)
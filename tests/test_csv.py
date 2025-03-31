import numpy as np
from pandas import read_csv
from sktalk.corpus.parsing.csv import *

def test_samplerate_from_key():
    test_cases = [
        ("/public-dutch/dutch-01", 24000),
        ("/public-spanish/spanish-01", 16000),
        ("/public-spanish/spanish-02", 16000),
        ("missing_file", 0),
    ]
    
    for key, expected_rate in test_cases:
        rate = samplerate_from_key(key)
        assert(rate == expected_rate)

def test_audio_from_key():
    ## These tests have been built by manually listening to the audio
    ## and ensuring the audio snippet is correct.
    ##
    ## In order to automate the tests, we use a checksum to compare
    ## obtained and expected results.
    test_cases = [
        ("/public-dutch/dutch-01", np.float32(0.019592285)),
        ("/public-spanish/spanish-01", np.float32(-0.042541504)),
        ("/public-spanish/spanish-02", np.float32(0.004272461)),
        ("missing_file", None),
    ]

    for key, expected_checksum in test_cases:
        audio = audio_from_key(key)
        if expected_checksum is None:
            assert(audio == [None])
        else:
            checksum = np.sum(audio)
            assert(checksum == expected_checksum)

def test_extend_dataframe():
    df = read_csv("tests/testdata/test_csv.csv")
    df = extend_dataframe(df)

    expected_values = {
        "key": ["/public-dutch/dutch-01", "/public-spanish/spanish-01", "/public-spanish/spanish-02", "/missing_file", "/public-spanish/spanish-wrong"],
        "rate": [24000, 16000, 16000, 0, 16000],
    }
    for column, expected in expected_values.items():
        assert(list(df[column]) == expected)
    
    ## Check that audio is present if available
    assert([len(df["audio"][i]) for i in range(len(df))] == [21600, 20800, 16000, 0, 0])


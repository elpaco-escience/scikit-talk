import numpy as np
from pandas import read_csv
from sktalk.corpus.parsing.csv import *

def test_samplerate_from_key():
    rate = samplerate_from_key("/public-dutch/dutch-01")
    assert(rate == 24000)

    rate = samplerate_from_key("/public-spanish/spanish-01")
    assert(rate == 16000)

    rate = samplerate_from_key("/public-spanish/spanish-02")
    assert(rate == 16000)

    rate = samplerate_from_key("missing_file")
    assert(rate == 0)

def test_audio_from_key():
    ## These tests have been built by manually listening to the audio
    ## and ensuring the audio snippet is correct.
    ##
    ## In order to automate the tests, we use a checksum to compare
    ## obtained and expected results.
    audio = audio_from_key("/public-dutch/dutch-01")
    checksum = np.sum(audio)
    assert(checksum == np.float32(0.019592285))

    audio = audio_from_key("/public-spanish/spanish-01")
    checksum = np.sum(audio)
    assert(checksum == np.float32(-0.042541504))

    audio = audio_from_key("/public-spanish/spanish-02")
    checksum = np.sum(audio)
    assert(checksum == np.float32(0.004272461))

    audio = audio_from_key("missing_file")
    assert(audio == [None])

def test_extend_dataframe():
    df = read_csv("tests/testdata/test_csv.csv")
    df = extend_dataframe(df)

    assert(list(df["key"]) == ["/public-dutch/dutch-01", "/public-spanish/spanish-01", "/public-spanish/spanish-02", "/missing_file", "/public-spanish/spanish-wrong"])
    assert(list(df["rate"]) == [24000, 16000, 16000, 0, 16000])
    
    ## Check that audio is present if available
    assert([len(df["audio"][i]) for i in range(len(df))] == [21600, 20800, 16000, 0, 0])


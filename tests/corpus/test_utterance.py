import pytest
from sktalk.corpus.utterance import Utterance


class TestUtterance():
    def test_init(self):
        # init with some basic data
        utterance = Utterance(
            utterance="Hello",
            participant="A",
            begin="00:00:00.000",
            end="00:00:02.856",
        )
        assert utterance.utterance == "Hello"
        assert utterance.participant == "A"
        assert utterance.begin == "00:00:00.000"
        assert utterance.end == "00:00:02.856"
        assert utterance.metadata is None

    def test_empty_init(self):
        with pytest.raises(TypeError):
            Utterance()  # noqa no-value-for-parameter

    def test_get_audio(self, utterances_with_time):
        utt = utterances_with_time[0]
        utt.get_audio()
        # assert utt.audio is not None
        # assertions about the type of returned audio
        # assert utt.audio.shape == (2856, 1)
        # assert utt.audio.dtype == np.float32

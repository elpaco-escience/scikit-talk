import pytest
from sktalk.corpus.utterance import Utterance


class TestUtterance():
    @pytest.mark.parametrize("utt_in, nwords, nchars", [
        ("Hello world", 2, 10),
        ("One", 1, 3),
        ("", 0, 0),
        ("Hello [laugh]", 1, 5),
        ("[laugh] hello [laugh] [noise]!", 1, 5),
        ("Hello 567 world", 2, 10),
        ("He5lo wor4d", 2, 10),
        ("zung1 ji3 jyut6", 3, 13),
        ("我 我 是 上学 去, 我 现在 给 她 买 diaper@s 了 .", 12, 20)
    ])
    def test_postinit(self, utt_in, nwords, nchars):
        utt = Utterance(
            utterance=utt_in
        )
        assert utt.n_words == nwords
        assert utt.n_characters == nchars

    def test_asdict(self):
        utt = Utterance(
            utterance="Hello world"
        )
        utt_dict = utt.asdict()
        assert utt_dict["utterance"] == "Hello world"
        assert utt_dict["n_words"] == 2

    def test_until(self, convo_utts):
        utt1, utt2 = convo_utts
        assert utt1.until(utt2) == -100

    milliseconds_timestamp = [
        ["0", "00:00:00.000"],
        ["1706326", "00:28:26.326"],
        ["222222", "00:03:42.222"],
        ["None", None]
    ]

    @pytest.mark.parametrize("milliseconds, timestamp", milliseconds_timestamp)
    def test_to_timestamp(self, milliseconds, timestamp):
        utt = Utterance(utterance="")
        assert utt._to_timestamp(milliseconds) == timestamp   # noqa: W0212

    def test_to_timestamp_errors(self):
        utt = Utterance(utterance="")
        with pytest.raises(ValueError, match="exceeds 24h"):
            utt._to_timestamp("987654321")                    # noqa: W0212

        with pytest.raises(ValueError, match="negative"):
            utt._to_timestamp("-1")                           # noqa: W0212

    time_begin_end = [[(1748070, 1751978), "00:29:08.070", "00:29:11.978"],
                      [[1748070, 1751978], "00:29:08.070", "00:29:11.978"],
                      [[1], None, None],
                      [1, None, None],
                      [None, None, None]]

    @pytest.mark.parametrize("time, begin, end", time_begin_end)
    def test_split_time(self, time, begin, end):
        utt = Utterance(utterance="", time=time)
        assert utt.begin == begin
        assert utt.end == end

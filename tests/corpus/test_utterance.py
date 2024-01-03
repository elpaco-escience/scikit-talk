import pytest
from sktalk.corpus.utterance import Utterance


class TestUtterance():
    @pytest.mark.parametrize("utt_in, nwords, nchars, uttlist", [
        ("Hello world", 2, 10, ["Hello", "world"]),
        ("One", 1, 3, ["One"]),
        ("", 0, 0, []),
        ("Hello [laugh]", 1, 5,  ["Hello"]),
        ("[laugh] hello [laugh] [noise]!", 1, 5, ["hello"]),
        ("Hello 567 world", 2, 10, ["Hello", "world"]),
        ("He5lo wor4d", 2, 10, ["He5lo", "wor4d"]),
        ("zung1 ji3 jyut6", 3, 13, ["zung1", "ji3", "jyut6"]),
        ("上学 去, 我 现在 diaper@s 了 .", 6, 14,
         ["上学", "去", "我", "现在", "diapers", "了"])
    ])
    def test_postinit(self, utt_in, nwords, nchars, uttlist):
        utt = Utterance(
            utterance=utt_in
        )
        assert utt.n_words == nwords
        assert utt.n_characters == nchars
        assert utt.utterance_list == uttlist

    def test_asdict(self):
        utt = Utterance(
            utterance="Hello world"
        )
        utt_dict = utt.asdict()
        assert utt_dict["utterance"] == "Hello world"
        assert utt_dict["n_words"] == 2

    def test_until(self, convo_utts):
        utt1, utt2 = convo_utts[:2]
        assert utt1.until(utt2) == -100

    @pytest.mark.parametrize("indices, expected", [
        ([0, 1], False),
        ([0, 2], True),
        ([0, 7], None),
        ([1, 7], None),
        ([7, 7], None)
    ])
    def test_same_speaker(self, convo_utts, indices, expected):
        utt1, utt2 = [convo_utts[i] for i in indices]
        assert utt1.same_speaker(utt2) == expected

    @pytest.mark.parametrize("utterance_time, window, expected_percentage", [
        ([100, 200], [100, 200], 100),
        ([100, 200], [150, 250], 50),
        ([100, 200], [0, 400], 100),
        ([0, 400], [100, 200], 25),
        (None, [0, 300], None),
        ([100, 200], None, None),
        ([100, 200], [400, 500], 0)
    ])
    def test_window_overlap_percentage(self, utterance_time, window, expected_percentage):
        utterance = Utterance(utterance="text", time=utterance_time)
        assert utterance.window_overlap_percentage(
            window) == expected_percentage

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

import pytest
from sktalk.corpus.utterance import Utterance
from contextlib import nullcontext as does_not_raise


class TestUtterance():
    @pytest.mark.parametrize("utt_in, utt_out, nwords, nchars, uttlist", [
        ("Hello world", "Hello world", 2, 10, ["Hello", "world"]),
        ("One", "One", 1, 3, ["One"]),
        ("", "", 0, 0, []),
        ("Hello [laugh]", "Hello", 1, 5,  ["Hello"]),
        ("Hello <laugh>", "Hello", 1, 5,  ["Hello"]),
        ("Hello (3.1)", "Hello", 1, 5,  ["Hello"]),
        ("Hello (3.1) world", "Hello world", 2, 10,  ["Hello", "world"]),
        ("(3.1) world", "world", 1, 5,  ["world"]),
        ("[laugh] hello [laugh] [noise]!", "hello", 1, 5, ["hello"]),
        ("Hello 567 world", "Hello world", 2, 10, ["Hello", "world"]),
        ("He5lo wor4d", "He5lo wor4d", 2, 10, ["He5lo", "wor4d"]),
        ("zung1 ji3 jyut6", "zung1 ji3 jyut6",
         3, 13, ["zung1", "ji3", "jyut6"]),
        ("上学 去, 我 现在 diaper@s 了 .", "上学 去 我 现在 diapers 了", 6, 14,
         ["上学", "去", "我", "现在", "diapers", "了"])
    ])
    def test_postinit_utterance(self, utt_in, utt_out, nwords, nchars, uttlist):    # noqa: too-many-arguments
        utt = Utterance(
            utterance=utt_in
        )
        assert utt.utterance == utt_out
        assert utt.n_words == nwords
        assert utt.n_characters == nchars
        assert utt.utterance_list == uttlist

    @pytest.mark.parametrize("time_in, time_out, timestamp_begin, warning", [
        (None, None, None, does_not_raise()),
        ([222222, 400000], [222222, 400000], "00:03:42.222", does_not_raise()),
        ("Text", None, None, pytest.warns(match="invalid time")),
        ("[1,2,3]", None, None, pytest.warns(match="invalid time")),
        ("[222222, 43523500000]", None, None,
         pytest.warns(match="invalid time")),
        ("[-4, 10]", None, None, pytest.warns(match="invalid time")),
        ("[29, 10]", None, None, pytest.warns(match="invalid time")),
        ("['text', 'text']", None, None, pytest.warns(match="invalid time"))
    ])
    def test_postinit_time(self, time_in, time_out, timestamp_begin, warning):
        with warning:
            utt = Utterance(
                utterance="text",
                time=time_in
            )
            assert utt.time == time_out
            assert utt.begin_timestamp == timestamp_begin

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
        [0, "00:00:00.000"],
        [1706326, "00:28:26.326"],
        [222222, "00:03:42.222"]
    ]

    @pytest.mark.parametrize("milliseconds, timestamp", milliseconds_timestamp)
    def test_to_timestamp(self, milliseconds, timestamp):
        utt = Utterance(utterance="")
        assert utt._to_timestamp(milliseconds) == timestamp   # noqa: W0212

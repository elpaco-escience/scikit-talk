import os
import tempfile
import pytest
import requests
import sktalk
from sktalk.corpus.parsing.cha import ChaParser
from sktalk.corpus.parsing.parser import Parser


class TestParser:
    milliseconds_timestamp = [
        ["0", "00:00:00.000"],
        ["1706326", "00:28:26.326"],
        ["222222", "00:03:42.222"],
        ["None", None]
    ]

    @pytest.mark.parametrize("milliseconds, timestamp", milliseconds_timestamp)
    def test_to_timestamp(self, milliseconds, timestamp):
        assert Parser._to_timestamp(milliseconds) == timestamp

        with pytest.raises(ValueError, match="exceeds 24h"):
            Parser._to_timestamp("987654321")

        with pytest.raises(ValueError, match="negative"):
            Parser._to_timestamp("-1")


class TestChaParser:
    URLs = [
        "https://ca.talkbank.org/data-orig/GCSAusE/01.cha",
        "https://ca.talkbank.org/data-orig/GCSAusE/02.cha",
        "https://ca.talkbank.org/data-orig/GCSAusE/03.cha"
    ]

    @pytest.fixture(params=URLs)
    def download_file(self, request):
        remote = request.param
        response = requests.get(remote)
        response.raise_for_status()

        ext = os.path.splitext(remote)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        yield temp_file_path

        os.remove(temp_file_path)

    @pytest.mark.parametrize("download_file", URLs, indirect=True)
    def test_parse(self, download_file):
        parsed_cha = ChaParser().parse(download_file)
        assert isinstance(parsed_cha, sktalk.corpus.conversation.Conversation)
        source = parsed_cha.utterances[0].source
        assert os.path.splitext(source)[1] == ".cha"
        assert parsed_cha.utterances[0].begin == "00:00:00.000"
        participant = parsed_cha.utterances[0].participant
        assert participant in ["A", "B", "S"]
        language = parsed_cha.metadata["Languages"]
        assert language == ["eng"]
        # TODO assert that there are no empty utterances

    def test_split_time(self):
        time = "(1748070, 1751978)"
        begin_end = ("00:29:08.070", "00:29:11.978")
        assert ChaParser._split_time(time) == begin_end

        time = None
        begin_end = (None, None)
        assert ChaParser._split_time(time) == begin_end

    unclean_clean = [
        [
            r"{'SAM': 'que (0.5) e(u) gosto \x151790561_1793421\x15 (0.2)→'}",
            "que (0.5) e(u) gosto (0.2)→"
        ],
        [
            r"{'SOR': 'hm → \x151706328_1706744\x15'}",
            "hm →"
        ],
        [
            r"",
            ""
        ],
        [
            r"{'T': '- what \x15128_128\x15 just (0.3) (.) \x15136_236\x15'}",
            "- what just (0.3) (.)"
        ]
    ]

    @pytest.mark.parametrize("unclean, clean", unclean_clean)
    def test_clean_utterance(self, unclean, clean):
        assert ChaParser._clean_utterance(unclean) == clean

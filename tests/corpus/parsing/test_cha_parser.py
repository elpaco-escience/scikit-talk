import requests
import pytest
import tempfile
import os

import sktalk
from sktalk.corpus.parsing.parser import Parser
from sktalk.corpus.parsing.cha import ChaParser


class TestParser:
    milliseconds_timestamp = [
        ["0", "00:00:00.000"],
        ["1706326", "00:28:26.326"],
        ["222222", "00:03:42.222"],
        ["None", None]
    ]

    @pytest.mark.parametrize("milliseconds_timestamp", milliseconds_timestamp)
    def test_to_timestamp(self, milliseconds_timestamp):
        milliseconds, timestamp = milliseconds_timestamp
        assert Parser._to_timestamp(milliseconds) == timestamp

        with pytest.raises(ValueError, match="exceeds 24h"):
            Parser._to_timestamp("987654321")

        with pytest.raises(ValueError, match="negative"):
            Parser._to_timestamp("-1")


class TestChaParser:
    URLs = [
        'https://ca.talkbank.org/data-orig/GCSAusE/01.cha',
        'https://ca.talkbank.org/data-orig/GCSAusE/02.cha',
        'https://ca.talkbank.org/data-orig/GCSAusE/03.cha'
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
        # assert that there are no empty data fields

    def test_split_time(self):
        time_input = "(1748070, 1751978)"
        expected_output = ("00:29:08.070", "00:29:11.978")
        assert ChaParser._split_time(time_input) == expected_output

        time_input = None
        expected_output = (None, None)
        assert ChaParser._split_time(time_input) == expected_output

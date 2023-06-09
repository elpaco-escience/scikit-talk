import requests
import pytest
import tempfile
import os

import sktalk
from sktalk.corpus.parsing.cha import ChaParser


class TestChaParser:
    URLs = [
        'https://ca.talkbank.org/data-orig/GCSAusE/01.cha',
        'https://ca.talkbank.org/data-orig/GCSAusE/02.cha'
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

    @pytest.mark.parametrize('download_file', URLs, indirect=True)
    def test_parse(self, download_file):
        parsed_cha = ChaParser().parse(download_file)
        assert isinstance(parsed_cha, sktalk.corpus.conversation.Conversation)

import pytest
from sktalk.corpus.parsing.json import JsonFile
from sktalk.corpus.conversation import Conversation


def test_jsonfile_parse():
    json_in = JsonFile("tests/testdata/test_conversation.json").parse()
    assert isinstance(json_in, Conversation)
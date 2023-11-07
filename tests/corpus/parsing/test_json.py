import pytest
from sktalk.corpus.parsing.json import JsonFile
from sktalk.corpus.conversation import Conversation


def test_jsonfile_parse():
    json_in = JsonFile("tests/testdata/test_conversation.json").parse()
    assert isinstance(json_in, Conversation)
    assert len(json_in.utterances) == 3
    assert json_in.utterances[0].utterance == "Hello world"
    with pytest.raises(KeyError):
        json_in.metadata["Utterances"]
    assert json_in.metadata["Languages"] == ["eng"]
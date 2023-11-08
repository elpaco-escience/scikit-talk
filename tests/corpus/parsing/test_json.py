import pytest
from sktalk.corpus.conversation import Conversation
from sktalk.corpus.corpus import Corpus
from sktalk.corpus.parsing.json import JsonFile


def test_jsonfile_parse():
    json_in = JsonFile("tests/testdata/dummy_conversation.json").parse()
    assert isinstance(json_in, Conversation)
    assert len(json_in.utterances) == 3
    assert json_in.utterances[0].utterance == "Hello world"
    with pytest.raises(KeyError):
        json_in.metadata["Utterances"] # noqa pointless-statement
    assert json_in.metadata["Languages"] == ["eng"]

def test_jsonfile_parse_as_corpus():
    json_in = Corpus.from_json("tests/testdata/dummy_corpus.json")
    assert isinstance(json_in, Corpus)
    


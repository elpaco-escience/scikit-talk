import json
import os
import pytest
from sktalk.corpus.conversation import Conversation


class TestConversation:
    def test_instantiate(self, my_convo, convo_utts, convo_meta):
        # test the conversation fixture
        assert isinstance(my_convo, Conversation)
        # test instantiation of a new conversation with content
        new_convo = Conversation(utterances=convo_utts,
                                 metadata=convo_meta)
        assert isinstance(new_convo, Conversation)
        # cannot instantiate an empty conversation
        with pytest.raises(TypeError):
            Conversation()  # noqa no-value-for-parameter
        # A conversation without metadata still has metadata property
        new_convo = Conversation(utterances=convo_utts)
        assert new_convo.metadata == {}
        # A Conversation can't be instantiated with utterances not of Utterance class
        with pytest.raises(TypeError, match="type Utterance"):
            Conversation(utterances="Not an Utterance")
        with pytest.raises(TypeError, match="type Utterance"):
            Conversation(utterances=0)
        # The user should be warned if there are no Utterances
        with pytest.warns(match="no Utterances"):
            Conversation(utterances=[])

    def test_asdict(self, my_convo):
        """Verify content of dictionary based on conversation"""
        convodict = my_convo.asdict()
        assert isinstance(convodict, dict)
        assert convodict["Utterances"][0] == my_convo.utterances[0].asdict()
        assert convodict["source"] == my_convo.metadata["source"]
        assert isinstance(convodict["Utterances"][0], dict)

    @pytest.mark.parametrize("user_path, expected_path", [
        ("tmp_convo.json", "tmp_convo.json"),
        ("tmp_convo", "tmp_convo.json")
    ])
    def test_write_json(self, my_convo, tmp_path, user_path, expected_path):
        tmp_file = f"{str(tmp_path)}{os.sep}{user_path}"
        my_convo.write_json(tmp_file)
        tmp_file_exp = f"{str(tmp_path)}{os.sep}{expected_path}"
        assert os.path.exists(tmp_file_exp)
        with open(tmp_file_exp) as f:
            my_convo_read = json.load(f)
            assert isinstance(my_convo_read, dict)
            assert my_convo_read == my_convo.asdict()

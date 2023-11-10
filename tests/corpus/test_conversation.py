import json
import os
from contextlib import nullcontext as does_not_raise
import pytest
from sktalk.corpus.conversation import Conversation


class TestConversation:
    def test_instantiate(self, convo, convo_utts, convo_meta):
        # test the conversation fixture
        assert isinstance(convo, Conversation)
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

    def test_asdict(self, convo):
        """Verify content of dictionary based on conversation"""
        convodict = convo.asdict()
        assert convodict["Utterances"][0] == convo.utterances[0].asdict()
        assert convodict["source"] == convo.metadata["source"]

    @pytest.mark.parametrize("user_path, expected_path", [
        ("tmp_convo.json", "tmp_convo.json"),
        ("tmp_convo", "tmp_convo.json")
    ])
    def test_write_json(self, convo, tmp_path, user_path, expected_path):
        tmp_file = f"{str(tmp_path)}{os.sep}{user_path}"
        convo.write_json(tmp_file)
        tmp_file_exp = f"{str(tmp_path)}{os.sep}{expected_path}"
        assert os.path.exists(tmp_file_exp)
        with open(tmp_file_exp, encoding='utf-8') as f:
            convo_read = json.load(f)
            assert isinstance(convo_read, dict)
            assert convo_read == convo.asdict()


class TestConversationMetrics:
    @pytest.mark.parametrize("index, before, after, time_or_index, error",
                             [
                                 (0, 0, 1, "index", does_not_raise()),
                                 (0, 1, 1, "index", pytest.raises(IndexError)),
                                 (9, 1, 0, "index", does_not_raise()),
                                 (9, 1, 1, "index", pytest.raises(IndexError)),
                                 (0, 0, 0, "neither_time_nor_index",
                                     pytest.raises(ValueError))
                             ])
    def test_subconversation_errors(self, convo, index, before, after, time_or_index, error):
        with error:
            convo.subconversation(index=index,
                                  before=before,
                                  after=after,
                                  time_or_index=time_or_index)

    @pytest.mark.parametrize("index, before, after, time_or_index, expected_length",
                             [
                                 (0, 0, 1, "index", 2),
                                 (5, 2, 0, "index", 3),
                                 (1, 1000, 0, "time", 2),
                                 (5, 3000, 3000, "time", 7),
                             ])
    def test_subconversation(self, convo, index, before, after, time_or_index, expected_length):
        sub = convo.subconversation(index=index,
                                    before=before,
                                    after=after,
                                    time_or_index=time_or_index)
        assert isinstance(sub, Conversation)
        assert len(sub.utterances) == expected_length

    @pytest.mark.parametrize("index, before, after, time_or_index, expected",
                             [(0, 0, 1, "index", -100)])
    def test_until(self, convo, index, before, after, time_or_index, expected):
        assert convo.subconversation(index=index,
                                     before=before,
                                     after=after,
                                     time_or_index=time_or_index).until_next == expected

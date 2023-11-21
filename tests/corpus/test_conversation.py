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
    @pytest.mark.parametrize("args, error",
                             [
                                 ([0, 0, 1, "index"], does_not_raise()),
                                 ([20, 1, 1, "index"], pytest.raises(IndexError)),
                                 ([0, 50, 50, "index"], does_not_raise()),
                                 ([0, 0, 0, "neither_time_nor_index"],
                                     pytest.raises(ValueError))
                             ])
    def test_subconversation_errors(self, convo, args, error):
        index, before, after, time_or_index = args
        with error:
            convo._subconversation(index=index,            #noqa protected-access
                                   before=before,
                                   after=after,
                                   time_or_index=time_or_index)

    @pytest.mark.parametrize("args, expected_length",
                             [
                                 ([0, 0, 1, "index"], 2),
                                 ([5, 2, 0, "index"], 3),
                                 ([0, 2, 2, "index"], 3),
                                 ([0, 2, None, "index"], 3),
                                 ([0, 0, 0, "time"], 2),  # A, B
                                 ([5, 3000, 3000, "time"], 6),  # B,C,E,U,F,H
                                 ([5, 0, 0, "time"], 3),  # C, U, F
                             ])
    def test_subconversation(self, convo, args, expected_length):
        index, before, after, time_or_index = args
        sub = convo._subconversation(index=index,           #noqa protected-access
                                     before=before,
                                     after=after,
                                     time_or_index=time_or_index)
        assert isinstance(sub, Conversation)
        assert len(sub.utterances) == expected_length

    def test_overlap(self):
        # entire utterance in window
        assert Conversation.overlap(80, 120, [90, 110])
        # beginning of utterance in window
        assert Conversation.overlap(80, 100, [90, 110])
        # end of utterance in window
        assert Conversation.overlap(100, 120, [90, 110])
        # utterance covers window entirely
        assert Conversation.overlap(95, 105, [90, 110])
        assert not Conversation.overlap(
            120, 140, [90, 110])  # utterance before window
        assert not Conversation.overlap(
            70, 80, [90, 110])  # utterance after window

    def test_count_participants(self, convo):
        assert convo.count_participants() == 3
        convo2 = convo._subconversation(index=0, before=2)  #noqa protected-access
        assert convo2.count_participants() == 2
        convo3 = convo._subconversation(index=0)            #noqa protected-access
        assert convo3.count_participants() == 1

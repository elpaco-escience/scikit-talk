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
                                 ([0, 0, 1], does_not_raise()),
                                 ([20, 1, 1], pytest.raises(IndexError)),
                                 ([0, 50, 50], does_not_raise())
                             ])
    def test_subconversation_errors(self, convo, args, error):
        index, before, after = args
        with error:
            convo._subconversation_by_index(index=index,            # noqa W0212
                                   before=before,
                                   after=after)

    @pytest.mark.parametrize("args, expected_length",
                             [
                                 ([0, 0, 1], 2),
                                 ([5, 2, 0], 3),
                                 ([0, 2, 2], 3),
                                 ([0, 2, None], 3)
                             ])
    def test_subconversation_index(self, convo, args, expected_length):
        index, before, after = args
        sub = convo._subconversation_by_index(index=index,           # noqa W0212
                                     before=before,
                                     after=after)
        assert isinstance(sub, Conversation)
        assert len(sub.utterances) == expected_length

    @pytest.mark.parametrize("args, expected_length",
                             [
                                 ([0, 0, 0, False], 2),  # A, B
                                 ([5, 3000, 3000, False], 8),  # B-H
                                 ([5, 0, 0, False], 5),  # C-F
                                 # no time window, only return U
                                 ([5, 0, 0, True], 1),
                                 # 7 has no timing
                                 ([7, 1000, 1000, False], 0),
                                 ([5, 0, 1500, True], 4),  # U-H
                                 ([5, 1000, 0, True], 4),  # C-U
                             ])
    def test_subconversation_time(self, convo, args, expected_length):
        index, before, after, exclude = args
        sub = convo._subconversation_by_time(index=index,           # noqa W0212
                                     before=before,
                                     after=after,
                                     exclude_utterance_overlap=exclude
                                     )
        assert isinstance(sub, Conversation)
        assert len(sub.utterances) == expected_length

    def test_count_participants(self, convo):
        assert convo.count_participants() == 4
        assert convo.count_participants(except_none=True) == 3
        convo2 = convo._subconversation_by_index(index=0, before=2)  # noqa W0212
        assert convo2.count_participants() == 2
        convo3 = convo._subconversation_by_index(index=0)            # noqa W0212
        assert convo3.count_participants() == 1

    def test_calculate_FTO(self, convo):
        convo.calculate_FTO()
        assert convo.metadata["Calculations"]["FTO"] == {
            "window": 10000, "planning_buffer": 200, "n_participants": 2}
        convo.calculate_FTO(window=10)
        assert convo.metadata["Calculations"]["FTO"] == {
            "window": 10, "planning_buffer": 200, "n_participants": 2}
        assert convo.utterances[0].FTO is None
        assert convo.utterances[1].FTO == -100
        assert convo.utterances[2].FTO is None
        convo.calculate_FTO(planning_buffer=0)
        assert convo.utterances[2].FTO == -2499

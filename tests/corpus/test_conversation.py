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
        assert new_convo.metadata == {"source": "unknown"}
        # A Conversation can't be instantiated with utterances not of Utterance class
        with pytest.raises(TypeError, match="type Utterance"):
            Conversation(utterances="Not an Utterance")
        with pytest.raises(TypeError, match="type Utterance"):
            Conversation(utterances=0)
        # The user should be warned if there are no Utterances
        with pytest.warns(match="no Utterances"):
            Conversation(utterances=[])

    def test_from_jsonfile(self):
        json_in = Conversation.from_json(
            "tests/testdata/dummy_conversation.json")
        assert isinstance(json_in, Conversation)
        assert len(json_in.utterances) == 3
        assert json_in.utterances[0].utterance == "Hello world"
        with pytest.raises(KeyError):
            json_in.metadata["Utterances"]  # noqa pointless-statement
        assert json_in.metadata["Languages"] == ["eng"]
        with pytest.raises(TypeError, match="cannot be imported as a Conversation"):
            Conversation.from_json("tests/testdata/dummy_corpus.json")

    def test_conversation_properties(self, convo):
        assert convo.participants == {"A", "B", "C", None}
        assert convo.n_utterances == 10

    def test_conversation_selection(self, convo):
        selected_convo = convo.select(participant="A")
        assert selected_convo.participants == {"A"}
        assert selected_convo.n_utterances == 3
        selected_convo = convo.select(utterance="6 utterance F")
        assert selected_convo.n_utterances == 1
        assert selected_convo.utterances[0].utterance == "6 utterance F"
        selected_convo = convo.select()
        assert selected_convo.n_utterances == 10

    def test_conversation_summary(self, convo, capfd):
        convo.summary(n=1)
        captured = capfd.readouterr()
        assert captured.out.strip() == "(0 - 1000) A: '0 utterance A'"
        convo.summary(n=1, participant="C")
        captured = capfd.readouterr()
        assert captured.out.strip() == "(5500 - 7500) C: '6 utterance F'"


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

    @pytest.mark.parametrize("args, index, expected_fto",
                             [
                                 ([10000, 200, 2], 0, None),
                                 ([10000, 200, 2], 1, -800),  # from 1 to 0
                                 # no FTO possible in 1 person convo
                                 ([10000, 200, 1], 1, None),
                                 ([10000, 200, 2], 2, -600),  # from 2 to 0
                                 ([10000, 200, 2], 3, -400),  # from 3 to 0
                                 ([1, 200, 2], 3, -400),  # 0 still overlaps
                                 ([10000, 200, 2], 4, 100),  # from 4 to 0
                                 # 0 does not fit in window
                                 ([1, 200, 2], 4, None),
                                 # timing info missing
                                 ([10000, 200, 2], 5, None),
                                 # timing info on previous missing
                                 ([10000, 200, 2], 6, None),
                                 # utterance starts <200ms after prior
                                 ([10000, 200, 2], 7, None),
                                 # planning buffer adjusted
                                 ([10000, 100, 2], 7, -350),
                                 # missing participant
                                 ([10000, 200, 2], 8, None),
                                 # missing participant in previous
                                 ([10000, 200, 2], 9, None),
                                 ([100, 200, 2], 10, -100),  # fom 10 to 9
                                 # previous only has partial overlap
                                 ([400, 200, 2], 11, None)
                             ])
    def test_calculate_FTO(self, convo_fto, args, index, expected_fto):
        window, planning_buffer, n_participants = args
        convo_fto.calculate_FTO(window, planning_buffer, n_participants)

        # metadata is updated
        assert convo_fto.metadata["Calculations"]["FTO"] == {
            "window": window,
            "planning_buffer": planning_buffer,
            "n_participants": n_participants}

        # utterance fto is calculated correctly
        assert convo_fto.utterances[index].FTO == expected_fto

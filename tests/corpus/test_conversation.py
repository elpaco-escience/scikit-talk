import csv
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
        filename = f"{str(tmp_path)}{os.sep}{user_path}"
        convo.write_json(filename)
        filename_exp = f"{str(tmp_path)}{os.sep}{expected_path}"
        assert os.path.exists(filename_exp)
        with open(filename_exp, encoding='utf-8') as f:
            convo_read = json.load(f)
            assert isinstance(convo_read, dict)
            assert convo_read == convo.asdict()

    @pytest.mark.parametrize("user_path", [
        ("tmp.csv"),
        ("tmp.json"),
        ("tmp")
    ])
    def test_write_csv(self, user_path, convo, tmp_path):
        filename = f"{str(tmp_path)}{os.sep}{user_path}"
        convo.write_csv(filename)
        metadatapath = f"{str(tmp_path)}{os.sep}tmp_metadata.csv"
        assert os.path.exists(metadatapath)
        tmp_output_utterances = f"{str(tmp_path)}{os.sep}tmp_utterances.csv"
        assert os.path.exists(tmp_output_utterances)

    def test_write_csv_metadata(self, convo, tmp_path):
        filename = f"{str(tmp_path)}{os.sep}tmp.csv"
        convo.write_csv(filename)
        metadatapath = f"{str(tmp_path)}{os.sep}tmp_metadata.csv"
        with open(metadatapath, 'r', encoding="utf-8") as file:
            reader = csv.reader(file)
            csv_out = list(reader)

    @pytest.mark.parametrize("conversation, error", [
        ("convo", does_not_raise()),
        ("empty_convo", pytest.raises(FileNotFoundError))
    ])
    def test_write_csv_utterances(self, conversation, error, tmp_path, request):
        conversation = request.getfixturevalue(conversation)

        filename = f"{str(tmp_path)}{os.sep}tmp.csv"
        conversation.write_csv(filename)
        utterancepath = f"{str(tmp_path)}{os.sep}tmp_utterances.csv"

        with error:
            with open(utterancepath, 'r', encoding="utf-8") as file:
                reader = csv.reader(file)
                csv_out = list(reader)

            assert len(csv_out) == len(conversation.utterances)+1
            assert len(set(csv_out[0])) == len(csv_out[0])
            assert csv_out[0][0] == "conversation_ID"

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

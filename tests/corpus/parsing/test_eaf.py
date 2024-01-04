import pytest
from sktalk.corpus.parsing.eaf import EafFile


@pytest.fixture
def path_source():
    return "tests/testdata/file02.eaf"


@pytest.fixture
def eaf_info():
    n_utterances = 8
    utterance_first = "Ut enim ad minim veniam"
    utterance_last = "ullamco laboris nisi ut aliquip ex ea commodo consequat."
    participants = {'Aleph Alpha', 'Bet Beta'}
    timing = [
        [0, 820],
        [1420, 2020],
        [1420, 3860],
        [3880, 4480],
        [5800, 6200],
        [6600, 6860],
        [9540, 9660],
        [9540, 9660]]
    return n_utterances, utterance_first, utterance_last, participants, timing


@pytest.fixture
def expected_metadata():
    return {'source': 'tests/testdata/file02.eaf'}


class TestEafFile:
    def test_parse(self, path_source, eaf_info, expected_metadata):
        expected_n_utterances, expected_first, expected_last, expected_participants, expected_timing = eaf_info
        eaf_utts, eaf_meta = EafFile(path_source).parse()
        assert eaf_meta == expected_metadata
        assert len(eaf_utts) == expected_n_utterances
        assert eaf_utts[0].utterance == expected_first
        assert eaf_utts[-1].utterance == expected_last
        assert {u.participant for u in eaf_utts} == expected_participants
        parsed_timing = [utt.time for utt in eaf_utts]
        assert parsed_timing == expected_timing

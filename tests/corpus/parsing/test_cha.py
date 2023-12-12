import pytest
from sktalk.corpus.conversation import Conversation
from sktalk.corpus.parsing.cha import ChaFile


@pytest.fixture
def path_source():
    return "tests/testdata/file01.cha"


@pytest.fixture
def cha_info():
    n_utterances = 13
    participants = {'A', 'B'}
    return n_utterances, participants

@pytest.fixture
def expected_metadata():
    return {'source': 'tests/testdata/file01.cha',
            'UTF8': '',
            'PID': 'idsequence',
            'Languages': ['eng'],
            'Participants': {
                'A': {
                    'name': 'Ann',
                    'language': 'eng',
                    'corpus': 'test',
                    'age': '',
                    'sex': '',
                    'group': '',
                    'ses': '',
                    'role': 'Adult',
                    'education': '',
                    'custom': ''},
                'B': {
                    'name': 'Bert',
                    'language': 'eng',
                    'corpus': 'test',
                    'age': '',
                    'sex': '',
                    'group': '',
                    'ses': '',
                    'role': 'Adult',
                    'education': '',
                    'custom': ''}},
            'Options': 'CA',
            'Media': '01, audio'}


class TestChaFile:
    def test_parse(self, path_source, cha_info, expected_metadata):
        n_utterances, participants = cha_info
        cha_utts, cha_meta = ChaFile(path_source).parse()
        assert cha_meta == expected_metadata
        assert {u.participant for u in cha_utts} == participants
        assert len(cha_utts) == n_utterances

    def test_wrapped_parser(self, path_source, cha_info, expected_metadata):
        n_utterances, participants = cha_info
        parsed_cha = Conversation.from_cha(path_source)
        assert isinstance(parsed_cha, Conversation)
        assert parsed_cha.metadata == expected_metadata
        assert {u.participant for u in parsed_cha.utterances} == participants
        assert len(parsed_cha.utterances) == n_utterances

    unclean_clean = [
        [
            r"{'SAM': 'que (0.5) e(u) gosto \x151790561_1793421\x15 (0.2)→'}",
            "que (0.5) e(u) gosto (0.2)→"
        ],
        [
            r"{'SOR': 'hm → \x151706328_1706744\x15'}",
            "hm →"
        ],
        [
            r"",
            ""
        ],
        [
            r"{'T': '- what \x15128_128\x15 just (0.3) (.) \x15136_236\x15'}",
            "- what just (0.3) (.)"
        ]
    ]

    @pytest.mark.parametrize("unclean, clean", unclean_clean)
    def test_clean_utterance(self, unclean, clean):
        assert ChaFile._clean_utterance(unclean) == clean            # noqa: W0212

import pytest
from sktalk.corpus.conversation import Conversation
from sktalk.corpus.parsing.cha import ChaFile


class TestChaFile:
    def test_parse(self):
        path_source = "tests/testdata/file01.cha"
        parsed_cha = ChaFile(path_source).parse()
        assert isinstance(parsed_cha, Conversation)
        assert parsed_cha.metadata["source"] == path_source
        assert parsed_cha.utterances[0].begin == "00:00:00.000"
        all_participants = {u.participant for u in parsed_cha.utterances}
        assert all_participants == {'A', 'B'}
        language = parsed_cha.metadata["Languages"]
        assert language == ["eng"]
        assert len(parsed_cha.utterances) == 13

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

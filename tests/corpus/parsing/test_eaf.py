import pytest
from sktalk.corpus.parsing.eaf import EafFile


@pytest.fixture
def path_source():
    return "tests/testdata/file02.eaf"


@pytest.fixture
def eaf_info():
    n_utterances = 12
    utterance_first = "Ut enim ad minim veniam"
    utterance_last = "eu fugiat nulla pariatur."
    participants = {'Aleph Alpha', 'Bet Beta', 'A_Words'}
    timing = [
        [0, 820],
        [1420, 2020],
        [1420, 3860],
        [1420, 3860],
        [3880, 4480],
        [5800, 6200],
        [6600, 6860],
        [6600, 6860],
        [9540, 9660],
        [9540, 9660],
        [9540, 9660],
        [9540, 9660]]
    return n_utterances, utterance_first, utterance_last, participants, timing


@pytest.fixture
def expected_metadata():
    return {
        'source': 'tests/testdata/file02.eaf',
        'header': {
            'MEDIA_FILE': '',
            'TIME_UNITS': 'milliseconds'
        },
        'adocument': {
            'AUTHOR': '',
            'DATE': '2015-09-04T05:39:16+01:00',
            'VERSION': '3.0',
            'FORMAT': '3.0',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:noNamespaceSchemaLocation': 'http://www.mpi.nl/tools/elan/EAFv2.8.xsd'
        },
        'licenses': [],
        'locales': {
            'en': ('US', None)
        },
        'languages': {},
        'media_descriptors': [
            {
            'MEDIA_URL': 'file:///Users/file.mp4',
            'MIME_TYPE': 'video/mp4',
            'RELATIVE_MEDIA_URL': './file.mp4'
            },
            {
            'MEDIA_URL': 'file:///Users/file.wav',
            'MIME_TYPE': 'audio/x-wav',
            'RELATIVE_MEDIA_URL': './file.wav'
            }
        ],
        'properties': [
            ('lastUsedAnnotationId', '3533')
        ],
        'linked_file_descriptors': [],
        'constraints': {
            'Time_Subdivision': "Time subdivision of parent annotation's time interval, no time gaps allowed within this interval",
            'Symbolic_Subdivision': 'Symbolic subdivision of a parent annotation. Annotations refering to the same parent are ordered',
            'Symbolic_Association': '1-1 association with a parent annotation',
            'Included_In': "Time alignable annotations within the parent annotation's time interval, gaps are allowed"
        },
        'linguistic_types': {
            'default-lt': {
            'GRAPHIC_REFERENCES': 'false',
            'LINGUISTIC_TYPE_ID': 'default-lt',
            'TIME_ALIGNABLE': 'true'
            },
            'Phrases': {
            'GRAPHIC_REFERENCES': 'false',
            'LINGUISTIC_TYPE_ID': 'Phrases',
            'TIME_ALIGNABLE': 'true'
            },
            'Text': {
            'GRAPHIC_REFERENCES': 'false',
            'LINGUISTIC_TYPE_ID': 'Text',
            'TIME_ALIGNABLE': 'true'
            },
            'Words': {
            'CONSTRAINTS': 'Symbolic_Subdivision',
            'GRAPHIC_REFERENCES': 'false',
            'LINGUISTIC_TYPE_ID': 'Words',
            'TIME_ALIGNABLE': 'false'
            },
            'Note': {
            'CONSTRAINTS': 'Symbolic_Association',
            'GRAPHIC_REFERENCES': 'false',
            'LINGUISTIC_TYPE_ID': 'Note',
            'TIME_ALIGNABLE': 'false'
            }
        },
        'controlled_vocabularies': {},
        'external_refs': {},
        'lexicon_refs': {}
        }


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

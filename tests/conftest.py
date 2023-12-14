import pytest
from sktalk.corpus.conversation import Conversation
from sktalk.corpus.corpus import Corpus
from sktalk.corpus.utterance import Utterance


@pytest.fixture
def convo_meta():
    return {
        'source': 'file.cha',
        'Languages': ['eng', 'fra'],
        'Participants': {
            'A': {
                'name': 'Aone',
                'age': '37',
                'sex': 'M'},
            'B': {
                'name': 'Btwo',
                'age': '22',
                'sex': 'M'}
        }
    }


@pytest.fixture
def expected_csv_metadata():
    return [
        ["source", "Languages", "Participants_A_name", "Participants_A_age", "Participants_A_sex",
            "Participants_B_name", "Participants_B_age", "Participants_B_sex"],
        ["file.cha", "eng, fra", "Aone", "37", "M", "Btwo", "22", "M"]
    ]


@pytest.fixture
def convo_utts():
    return [
        Utterance(
            utterance="0 utterance A",
            participant="A",
            time=[0, 1000]
        ),
        Utterance(
            utterance="1 utterance B",
            participant="B",
            time=[900, 3500]
        ),
        Utterance(
            utterance="2 utterance C",
            participant="A",
            time=[1001, 8500]
        ),
        Utterance(
            utterance="3 utterance D",
            participant="B",
            time=[1200, 1999]
        ),
        Utterance(
            utterance="4 utterance E",
            participant="A",
            time=[3500, 4500]
        ),
        Utterance(
            utterance="5 utterance U",
            participant="B",
            time=[5000, 8000]
        ),
        Utterance(
            utterance="6 utterance F",
            participant="C",
            time=[5500, 7500]
        ),
        Utterance(
            utterance="7 utterance G",
            participant=None,
            time=None
        ),
        Utterance(
            utterance="8 utterance H",
            participant="B",
            time=[9000, 12500]
        ),
        Utterance(
            utterance="9 utterance I",
            participant="C",
            time=[12000, 13000]
        )
    ]


@pytest.fixture
def expected_csv_utterances():
    return [
        ["", "source", "utterance", "participant", "time"],
        ["0", "file.cha", "0 utterance A", "A", "[0, 1000]"],
        ["1", "file.cha", "1 utterance B", "B", "[900, 3500]"],
        ["2", "file.cha", "2 utterance C", "A", "[1001, 8500]"],
        ["3", "file.cha", "3 utterance D", "B", "[1200, 1999]"],
        ["4", "file.cha", "4 utterance E", "A", "[3500, 4500]"],
        ["5", "file.cha", "5 utterance U", "B", "[5000, 8000]"],
        ["6", "file.cha", "6 utterance F", "C", "[5500, 7500]"],
        ["7", "file.cha", "7 utterance G", "", ""],
        ["8", "file.cha", "8 utterance H", "B", "[9000, 12500]"],
        ["9", "file.cha", "9 utterance I", "C", "[12000, 13000]"]]


@pytest.fixture
def convo(convo_utts, convo_meta):
    return Conversation(convo_utts, convo_meta)


@pytest.fixture
def empty_convo(convo_meta):
    return Conversation([], convo_meta, suppress_warnings=True)


@pytest.fixture
def utterances_for_fto():
    return [
        Utterance(
            utterance="utt 0 - A",
            participant="A",
            time=[0, 1000]
        ),
        Utterance(
            utterance="utt 1 - B",
            participant="B",
            time=[200, 300]
        ),
        Utterance(
            utterance="utt 2 - B",
            participant="B",
            time=[400, 500]
        ),
        Utterance(
            utterance="utt 3 - B",
            participant="B",
            time=[600, 900]
        ),
        Utterance(
            utterance="utt 4 - B",
            participant="B",
            time=[1100, 1500]
        ),
        Utterance(
            utterance="utt 5 - A",
            participant="A",
            time=None
        ),
        Utterance(
            utterance="utt 6 - A",
            participant="A",
            time=[1300, 1800]
        ),
        Utterance(
            utterance="utt 7 - B",
            participant="B",
            time=[1450, 1800]
        ),
        Utterance(
            utterance="utt 8 - None",
            participant=None,
            time=[1850, 1900]
        ),
        Utterance(
            utterance="utt 9 - A",
            participant="A",
            time=[1900, 2300]
        ),
        Utterance(
            utterance="utt 10 - B",
            participant="B",
            time=[2200, 2400]
        ),
        Utterance(
            utterance="utt 11 - B",
            participant="B",
            time=[2450, 2600]
        ),
    ]


@pytest.fixture
def convo_fto(utterances_for_fto, convo_meta):
    return Conversation(utterances_for_fto, convo_meta)


@pytest.fixture
def my_corpus():
    return Corpus(language="French",
                  importer="John Doe",
                  collections=["IADV", "Callosum"])

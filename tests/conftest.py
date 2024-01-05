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
            utterance="X0 utterance A",
            participant="A",
            time=[0, 1000]
        ),
        Utterance(
            utterance="X1 utterance B",
            participant="B",
            time=[900, 3500]
        ),
        Utterance(
            utterance="X2 utterance C",
            participant="A",
            time=[1001, 8500]
        ),
        Utterance(
            utterance="X3 utterance D",
            participant="B",
            time=[1200, 1999]
        ),
        Utterance(
            utterance="X4 utterance E",
            participant="A",
            time=[3500, 4500]
        ),
        Utterance(
            utterance="X5 utterance U",
            participant="B",
            time=[5000, 8000]
        ),
        Utterance(
            utterance="X6 utterance F",
            participant="C",
            time=[5500, 7500]
        ),
        Utterance(
            utterance="X7 utterance G",
            participant=None,
            time=None
        ),
        Utterance(
            utterance="X8 utterance H",
            participant="B",
            time=[9000, 12500]
        ),
        Utterance(
            utterance="X9 utterance I",
            participant="C",
            time=[12000, 13000]
        )
    ]


@pytest.fixture
def expected_csv_utterances():
    return [
        ["", "source", "utterance", "participant", "time"],
        ["0", "file.cha", "X0 utterance A", "A", "[0, 1000]"],
        ["1", "file.cha", "X1 utterance B", "B", "[900, 3500]"],
        ["2", "file.cha", "X2 utterance C", "A", "[1001, 8500]"],
        ["3", "file.cha", "X3 utterance D", "B", "[1200, 1999]"],
        ["4", "file.cha", "X4 utterance E", "A", "[3500, 4500]"],
        ["5", "file.cha", "X5 utterance U", "B", "[5000, 8000]"],
        ["6", "file.cha", "X6 utterance F", "C", "[5500, 7500]"],
        ["7", "file.cha", "X7 utterance G", "", ""],
        ["8", "file.cha", "X8 utterance H", "B", "[9000, 12500]"],
        ["9", "file.cha", "X9 utterance I", "C", "[12000, 13000]"]]


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
            utterance="utt X0 - A",
            participant="A",
            time=[0, 1000]
        ),
        Utterance(
            utterance="utt X1 - B",
            participant="B",
            time=[200, 300]
        ),
        Utterance(
            utterance="utt X2 - B",
            participant="B",
            time=[400, 500]
        ),
        Utterance(
            utterance="utt X3 - B",
            participant="B",
            time=[600, 900]
        ),
        Utterance(
            utterance="utt X4 - B",
            participant="B",
            time=[1100, 1500]
        ),
        Utterance(
            utterance="utt X5 - A",
            participant="A",
            time=None
        ),
        Utterance(
            utterance="utt X6 - A",
            participant="A",
            time=[1300, 1800]
        ),
        Utterance(
            utterance="utt X7 - B",
            participant="B",
            time=[1450, 1800]
        ),
        Utterance(
            utterance="utt X8 - None",
            participant=None,
            time=[1850, 1900]
        ),
        Utterance(
            utterance="utt X9 - A",
            participant="A",
            time=[1900, 2300]
        ),
        Utterance(
            utterance="utt X10 - B",
            participant="B",
            time=[2200, 2400]
        ),
        Utterance(
            utterance="utt X11 - B",
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


@pytest.fixture
def my_corpus_with_convo(my_corpus, convo):
    c = my_corpus
    c.append(convo)
    c.append(convo)
    return c


@pytest.fixture
def expected_csv_metadata_corpus():
    return [
        ["language", "importer", "collections", "source", "Languages", "Participants_A_name", "Participants_A_age", "Participants_A_sex",
            "Participants_B_name", "Participants_B_age", "Participants_B_sex"],
        ["French", "John Doe", "IADV, Callosum", "file.cha",
            "eng, fra", "Aone", "37", "M", "Btwo", "22", "M"],
        ["French", "John Doe", "IADV, Callosum", "file.cha",
            "eng, fra", "Aone", "37", "M", "Btwo", "22", "M"]
    ]


@pytest.fixture
def expected_csv_utterances_corpus():
    return [
        ["", "source", "utterance", "participant", "time"],
        ["0", "file.cha", "X0 utterance A", "A", "[0, 1000]"],
        ["1", "file.cha", "X1 utterance B", "B", "[900, 3500]"],
        ["2", "file.cha", "X2 utterance C", "A", "[1001, 8500]"],
        ["3", "file.cha", "X3 utterance D", "B", "[1200, 1999]"],
        ["4", "file.cha", "X4 utterance E", "A", "[3500, 4500]"],
        ["5", "file.cha", "X5 utterance U", "B", "[5000, 8000]"],
        ["6", "file.cha", "X6 utterance F", "C", "[5500, 7500]"],
        ["7", "file.cha", "X7 utterance G", "", ""],
        ["8", "file.cha", "X8 utterance H", "B", "[9000, 12500]"],
        ["9", "file.cha", "X9 utterance I", "C", "[12000, 13000]"],
        ["10", "file.cha", "X0 utterance A", "A", "[0, 1000]"],
        ["11", "file.cha", "X1 utterance B", "B", "[900, 3500]"],
        ["12", "file.cha", "X2 utterance C", "A", "[1001, 8500]"],
        ["13", "file.cha", "X3 utterance D", "B", "[1200, 1999]"],
        ["14", "file.cha", "X4 utterance E", "A", "[3500, 4500]"],
        ["15", "file.cha", "X5 utterance U", "B", "[5000, 8000]"],
        ["16", "file.cha", "X6 utterance F", "C", "[5500, 7500]"],
        ["17", "file.cha", "X7 utterance G", "", ""],
        ["18", "file.cha", "X8 utterance H", "B", "[9000, 12500]"],
        ["19", "file.cha", "X9 utterance I", "C", "[12000, 13000]"]]

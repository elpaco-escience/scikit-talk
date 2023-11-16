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
def convo_utts():
    return [
        Utterance(
            utterance="Hello A",
            participant="A",
            time=[0, 1000]
        ),
        Utterance(
            utterance="Monde B",
            participant="B",
            time=[900, 3500]
        ),
        Utterance(
            utterance="Hello C",
            participant="A",
            time=[1001, 12000]
        ),
        Utterance(
            utterance="Monde D",
            participant="B",
            time=[1200, 1999]
        ),
        Utterance(
            utterance="Hello E",
            participant="A",
            time=[3500, 4500]
        ),
        Utterance(
            utterance="Utterance U",
            participant="B",
            time=[5000, 8000]
        ),
        Utterance(
            utterance="Monde F",
            participant="B",
            time=[5500, 7500]
        ),
        Utterance(
            utterance="Hello G",
            participant="A",
            time=None
        ),
        Utterance(
            utterance="Monde H",
            participant="B",
            time=[9000, 12500]
        ),
        Utterance(
            utterance="Hello I",
            participant="C",
            time=[12000, 13000]
        )
    ]


@pytest.fixture
def convo(convo_utts, convo_meta):
    return Conversation(convo_utts, convo_meta)


@pytest.fixture
def my_corpus():
    return Corpus(language="French",
                  importer="John Doe",
                  collections=["IADV", "Callosum"])

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
    utterance1 = Utterance(
        utterance="Hello",
        participant="A",
        time=[0, 1000]
    )
    utterance2 = Utterance(
        utterance="Monde",
        participant="B",
        time=[900, 1800]
    )
    return [utterance1, utterance2]


@pytest.fixture
def my_convo(convo_utts, convo_meta):
    return Conversation(convo_utts, convo_meta)


@pytest.fixture
def my_corpus():
    return Corpus(language="French",
                  importer="John Doe",
                  collections=["IADV", "Callosum"])

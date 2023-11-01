import pytest
from sktalk.corpus.conversation import Conversation
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
def utt1():
    return {
        "utterance": "Hello",
        "participant": "A"
    }

@pytest.fixture
def utt2():
    return {
            "utterance": "Monde",
            "participant": "B"
        }

@pytest.fixture
def convo_utts(utt1, utt2):
    return [Utterance(utt1),
        Utterance(utt2)]

@pytest.fixture
def my_convo(convo_utts, convo_meta):
    return Conversation(convo_utts, convo_meta)
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
def convo_utts():
    return [Utterance({
        "utterance": "Hello",
        "participant": "A"
    }),
        Utterance({
            "utterance": "Monde",
            "participant": "B"
        })]

@pytest.fixture
def my_convo(convo_utts, convo_meta):
    return Conversation(convo_utts, convo_meta)
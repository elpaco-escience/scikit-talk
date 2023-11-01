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
    utterance1 = Utterance(
        utterance= "Hello",
        participant = "A"
    )
    utterance2 = Utterance(
        utterance = "Monde",
        participant = "B"
    )
    return [utterance1, utterance2]

@pytest.fixture
def my_convo(convo_utts, convo_meta):
    return Conversation(convo_utts, convo_meta)
import pytest
from sktalk.corpus.conversation import Conversation
from sktalk.corpus.utterance import Utterance



@pytest.fixture
def conversation_metadata():
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
def conversation_utterances():
    return [Utterance({
        "utterance": "Hello",
        "participant": "A"
    }),
        Utterance({
            "utterance": "Monde",
            "participant": "B"
        })]

@pytest.fixture
def my_convo(conversation_utterances, conversation_metadata):
    return Conversation(conversation_utterances, conversation_metadata)
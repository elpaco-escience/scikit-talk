import pytest
import os
from sktalk.corpus.conversation import Conversation
from sktalk.corpus.utterance import Utterance

@pytest.fixture
def my_convo():
    metadata = {
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

    utterances = [Utterance({
            "utterance": "Hello",
            "participant": "A"
        }),
        Utterance({
            "utterance": "Monde",
            "participant": "B"
        })]
    return Conversation(utterances, metadata)

class TestConversation:
    def test_instantiate(self, my_convo):
        assert isinstance(my_convo, Conversation)

    def test_write_json(self, my_convo):
        # Write a mock file and confirm that it has worked
        my_convo.write_json('file', '.')
        assert os.path.exists('file.jsonl')
        # Clean up
        os.remove('file.jsonl')

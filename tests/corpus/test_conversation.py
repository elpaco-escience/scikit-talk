import pytest
import os
from sktalk import Conversation

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

    utterances = [{
            "utterance": "Hello",
            "participant": "A"
        },
        {
            "utterance": "Monde",
            "participant": "B"
        }]
    return Conversation(utterances, metadata)

class TestConversation:
    def test_instantiate(self, my_convo):
        assert isinstance(my_convo, Conversation)

    def test_write_json(self, my_convo):
        # Write a mock file and confirm that it has worked
        my_convo.write_json('file.json', '.')
        assert os.path.exists('file.json')
        # Clean up
        os.remove('file.json')
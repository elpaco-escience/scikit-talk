import pytest
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

    def test_asdict(self, my_convo):
        """Verify content of dictionary based on conversation"""
        convodict = my_convo.asdict()
        assert isinstance(convodict, dict)
        assert convodict["Utterances"][0] == my_convo.utterances[0].asdict()
        assert convodict["source"] == my_convo.metadata["source"]

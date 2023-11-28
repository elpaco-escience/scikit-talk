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
        # time=(0, 2856),       TODO the tuple is written as list in json
        #                       therefore it is not equal to the original
        #                       object when opened with json.load()
        # see https://stackoverflow.com/questions/53804116/python3-can-json-be-decoded-as-tuple
        begin='00:00:00.000',
        end='00:00:02.856'
    )
    utterance2 = Utterance(
        utterance="Monde",
        participant="B",
        # time=(14511, 16961),
        begin='00:00:14.511',
        end='00:00:16.961'
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


@pytest.fixture
def utterances_with_time():
    utterance1 = Utterance(
        utterance="Hello",
        participant="A",
        time=(0, 2856),
        begin='00:00:00.000',
        end='00:00:02.856'
    )
    utterance2 = Utterance(
        utterance="Monde",
        participant="B",
        time=(14511, 16961),
        begin='00:00:14.511',
        end='00:00:16.961'
    )
    return [utterance1, utterance2]

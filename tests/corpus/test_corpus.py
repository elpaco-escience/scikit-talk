import pytest
from sktalk.corpus.corpus import Corpus

class TestCorpus():

    @pytest.mark.parametrize("conversations,metadata",
                             [
                                ([], {"author": "Person",
                                      "language": "french"}),
                                (None, {"author": "Person",
                                      "language": "french"}),
                                (None, {}),
                                ([], {})
                             ])
    def test_init(self, conversations, metadata):
        corpus: Corpus = Corpus(conversations = conversations,
                                **metadata)

        # check that a corpus object is created
        assert isinstance(corpus, Corpus)
        # corpus object has metadata (can be empty, but should be created)
        assert isinstance(corpus.metadata, dict)

        # corpus object has or has not (zero, one, multiple) conversations
        # assert corpus.conversations == conversations
        assert isinstance(corpus.conversations, list)
        assert conversations is None or corpus.conversations == conversations


        # initialize a corpus with conversation object that is not a conversation should fail


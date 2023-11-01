from contextlib import nullcontext as does_not_raise
import pytest
from sktalk.corpus.corpus import Corpus


class TestCorpus():

    @pytest.mark.parametrize("conversations,metadata,error",
                             [
                                ([], {"author": "Person",
                                      "language": "french"},
                                      does_not_raise()),
                                (None, {"author": "Person",
                                      "language": "french"},
                                      does_not_raise()),
                                (None, {}, does_not_raise()),
                                ([], {}, does_not_raise()),
                                ("Not A Conversation", {}, pytest.raises(TypeError))
                             ])
    def test_init(self, conversations, metadata, error):
        with error:
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

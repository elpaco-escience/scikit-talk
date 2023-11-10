import json
import os
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
                                 ("Not A Conversation", {},
                                     pytest.raises(TypeError))
                             ])
    def test_init(self, conversations, metadata, error):
        with error:
            corpus: Corpus = Corpus(conversations=conversations,
                                    **metadata)

            # check that a corpus object is created
            assert isinstance(corpus, Corpus)
            # corpus object has metadata (can be empty, but should be created)
            assert isinstance(corpus.metadata, dict)

            # corpus object has or has not (zero, one, multiple) conversations
            assert isinstance(corpus.conversations, list)
            assert conversations is None or corpus.conversations == conversations

    def test_append(self, my_corpus, convo):
        # conversation can be added to an existing corpus
        my_corpus.append(convo)
        assert my_corpus.conversations[-1] == convo

        # it is not possible to add non-conversation objects to a corpus
        with pytest.raises(TypeError, match="type Conversation"):
            my_corpus.append("Not A Conversation")

    def test_asdict(self, my_corpus):
        """Verify content of dictionary based on corpus"""
        corpusdict = my_corpus.asdict()
        assert isinstance(corpusdict, dict)
        assert corpusdict["language"] == my_corpus.metadata["language"]
        assert corpusdict["importer"] == my_corpus.metadata["importer"]

    @pytest.mark.parametrize("user_path, expected_path", [
        ("tmp_convo.json", "tmp_convo.json"),
        ("tmp_convo", "tmp_convo.json")
    ])
    def test_write_json(self, my_corpus, tmp_path, user_path, expected_path):
        tmp_file = f"{str(tmp_path)}{os.sep}{user_path}"
        my_corpus.write_json(tmp_file)
        tmp_file_exp = f"{str(tmp_path)}{os.sep}{expected_path}"
        assert os.path.exists(tmp_file_exp)
        with open(tmp_file_exp, encoding='utf-8') as f:
            my_corpus_read = json.load(f)
            assert isinstance(my_corpus_read, dict)
            assert my_corpus_read == my_corpus.asdict()

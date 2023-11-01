from sktalk.corpus.conversation import Conversation

class TestConversation:
    def test_instantiate(self, my_convo, convo_utts, convo_meta):
        assert isinstance(my_convo, Conversation)
        new_convo = Conversation(utterances = convo_utts,
                                 metadata = convo_meta)
        assert isinstance(new_convo, Conversation)

    def test_asdict(self, my_convo):
        """Verify content of dictionary based on conversation"""
        convodict = my_convo.asdict()
        assert isinstance(convodict, dict)
        assert convodict["Utterances"][0] == my_convo.utterances[0].asdict()
        assert convodict["source"] == my_convo.metadata["source"]
        assert isinstance(convodict["Utterances"][0], dict)

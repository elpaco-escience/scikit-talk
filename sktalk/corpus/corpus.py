from .parsing.json import JsonFile
from .parsing.xml import XmlFile
from sktalk.corpus.conversation import Conversation


class Corpus:
    def __init__(
        self, conversations: list["Conversation"] = None, **metadata: dict  # noqa: F821
    ):
        self._conversations = conversations or []
        for conversation in self._conversations:
            if not isinstance(conversation, Conversation):
                raise TypeError(
                    "All conversations should be of type Conversation")
        self._metadata = metadata

    def __add__(self, other: "Corpus") -> "Corpus":
        pass

    def append(self, conversation: Conversation):
        """
        Append a conversation to the Corpus

        Args:
            conversation (Conversation): Conversation object that should be added to the Corpus
        """
        if isinstance(conversation, Conversation):
            self._conversations.append(conversation)
        else:
            raise TypeError(
                "Conversations added should be of type Conversation")

    def return_json(self):
        self.return_dataframe()
        self.json = self.df.to_json()

    @property
    def metadata(self):
        """
        Get the metadata associated with the Corpus.

        Returns:
            dict: Additional metadata associated with the Corpus.
        """
        return self._metadata

    @property
    def conversations(self):
        """
        Get the conversations contained in the Corpus

        Returns:
            list: listed conversations contained in this Corpus
        """
        return self._conversations

    @classmethod
    def from_json(cls, path):
        return JsonFile(path).parse()

    @classmethod
    def from_xml(cls, path):
        return XmlFile(path).parse()

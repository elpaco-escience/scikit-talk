import json
from .conversation import Conversation
from .parsing.json import JsonFile
from .parsing.xml import XmlFile
from .write.writer import Writer


class Corpus(Writer):
    def __init__(
        self, conversations: list["Conversation"] = None, **metadata  # noqa: F821
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

    def asdict(self):
        """
        Return the Corpus as a dictionary

        Returns:
            dict: dictionary containing Corpus metadata and Conversations
        """
        return self._metadata | {"Conversations": [u.asdict() for u in self._conversations]}

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
        """Parse conversation file in JSON format

        Returns:
            Conversation: A Conversation object representing the conversation in the file.
        """
        with open(path, encoding='utf-8') as f:
            json_in = json.load(f)
        return Corpus(**json_in)

    @classmethod
    def from_xml(cls, path):
        return XmlFile(path).parse()

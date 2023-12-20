import json
import pandas as pd
from .conversation import Conversation
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

        self._metadata_df = None
        self._utterance_df = None

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
        """Parse corpus file in JSON format

        Returns:
            Corpus: A Corpus object representing the corpus in the file.
        """
        with open(path, encoding='utf-8') as f:
            json_in = json.load(f)
        return cls._fromdict(json_in)

    @classmethod
    def _fromdict(cls, fields):
        try:
            conversations = [Conversation._fromdict(
                c) for c in fields["Conversations"]]
            del fields["Conversations"]
        except KeyError as e:
            raise TypeError("This file cannot be imported as a Corpus.") from e
        return Corpus(conversations, metadata=fields)

    @classmethod
    def from_xml(cls, path):
        return XmlFile(path).parse()

    @property
    def metadata_df(self):
        """Return the corpus metadata as a pandas dataframe."""
        if self._metadata_df is None:
            metadata_df = self._metadata_to_df(self._metadata)
            metadata_df_conversations = pd.concat(
                [c.metadata_df for c in self._conversations])
            self._metadata_df = metadata_df.merge(
                metadata_df_conversations, how="cross")
        return self._metadata_df

    @property
    def utterance_df(self):
        """Return the corpus utterances as a pandas dataframe."""
        if self._utterance_df is None:
            self._utterance_df = pd.concat(
                [c.utterance_df for c in self._conversations], ignore_index=True)
        return self._utterance_df

from .parsing.json import JsonFile
from .parsing.xml import XmlFile


class Corpus:
    def __init__(
        self, conversations: list["Conversation"] = None, **metadata: dict  # noqa: F821
    ):
        self._conversations = conversations or []
        self._metadata = metadata

    def __add__(self, other: "Corpus") -> "Corpus":
        pass

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

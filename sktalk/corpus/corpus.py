from .conversation import Conversation
from .parsing.xml import XmlParser
from .parsing.json import JsonParser


class Corpus:
    def __init__(self, conversations: list[Conversation], metadata: dict):
        self._conversations = conversations
        self._metadata = metadata

    def __add__(self, other: "Corpus") -> "Corpus":
        pass

    def return_json(self):
        self.return_dataframe()
        self.json = self.df.to_json()

    @classmethod
    def from_json(cls, file):
        return JsonParser().parse(file)

    @classmethod
    def from_xml(cls, file):
        return XmlParser().parse(file)

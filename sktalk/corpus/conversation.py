from .parsing.xml import XmlParser
from .utterance import Utterance


class Conversation:
    def __init__(self, utterances: list[Utterance], metadata: dict) -> None:
        self._utterances = utterances
        self._metadata = metadata

    @property
    def utterances(self):
        return self._utterances

    @property
    def metadata(self):
        return self._metadata

    def get_utterance(self, index) -> Utterance:
        pass

    @classmethod
    def from_xml(cls, file):
        # read xml to dataframe
        return XmlParser.parse(file)

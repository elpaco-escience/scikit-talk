import abc

from ..conversation import Conversation
from ..corpus import Corpus


class Parser(abc.ABC):
    """Abstract parser class."""

    @abc.abstractmethod
    def parse(self, file) -> Conversation:
        return NotImplemented

    @abc.abstractmethod
    def parse_corpus(self, files) -> Corpus:
        return NotImplemented

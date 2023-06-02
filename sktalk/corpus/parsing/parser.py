import abc
from sktalk.corpus.conversation import Conversation
from sktalk.corpus.corpus import Corpus


class Parser(abc.ABC):
    """Abstract parser class."""

    @abc.abstractmethod
    def parse(self, file) -> Conversation:
        return NotImplemented

    @abc.abstractmethod
    def parse_corpus(self, files) -> Corpus:
        [self.parse(file) for file in files]

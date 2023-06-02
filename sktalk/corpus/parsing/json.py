from ..conversation import Conversation
from ..corpus import Corpus
from .parser import Parser


class JsonParser(Parser):
    def parse(self, file) -> Conversation:
        raise NotImplementedError()

    def parse_corpus(self, files) -> Corpus:
        raise NotImplementedError()

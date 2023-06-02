from ..conversation import Conversation
from ..corpus import Corpus
from .parser import Parser


class CsvParser(Parser):
    def parse(self, file) -> "Conversation":
        raise NotImplementedError()

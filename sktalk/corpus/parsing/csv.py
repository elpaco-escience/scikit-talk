from .parser import Parser


class CsvParser(Parser):
    def parse(self, file) -> "Conversation":  # noqa: F821
        raise NotImplementedError()

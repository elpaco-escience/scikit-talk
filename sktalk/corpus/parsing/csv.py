from .parser import InputFile


class CsvParser(InputFile):
    def parse(self, file) -> "Conversation":  # noqa: F821
        raise NotImplementedError()

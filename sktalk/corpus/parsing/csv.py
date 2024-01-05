from .parser import InputFile


class CsvFile(InputFile):
    def parse(self) -> "Conversation":  # noqa: F821
        raise NotImplementedError()

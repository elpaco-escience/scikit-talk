from .parser import InputFile


class JsonParser(InputFile):
    def parse(self, file) -> "Conversation":  # noqa: F821
        raise NotImplementedError()

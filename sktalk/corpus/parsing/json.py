from .parser import InputFile


class JsonFile(InputFile):
    def parse(self) -> "Conversation":  # noqa: F821
        raise NotImplementedError()

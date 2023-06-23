from .parser import InputFile


class XmlParser(InputFile):
    def parse(self, file) -> "Conversation":  # noqa: F821
        raise NotImplementedError()

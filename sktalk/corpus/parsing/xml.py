from .parser import InputFile


class XmlFile(InputFile):
    def parse(self) -> "Conversation":  # noqa: F821
        raise NotImplementedError()

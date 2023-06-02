from .parser import Parser


class JsonParser(Parser):
    def parse(self, file) -> "Conversation":  # noqa: F821
        raise NotImplementedError()

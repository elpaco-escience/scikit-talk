from .parser import Parser


class JsonParser(Parser):
    def parse(self, file) -> "Conversation":  # noqa: F821
        raise NotImplementedError()

    def parse_corpus(self, files) -> "Corpus":  # noqa: F821
        raise NotImplementedError()

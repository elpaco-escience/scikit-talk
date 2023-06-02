from .parser import Parser


class XmlParser(Parser):
    def parse(self, file) -> "Conversation":  # noqa: F821
        raise NotImplementedError()

    def parse_corpus(self, files) -> "Corpus":  # noqa: F821
        raise NotImplementedError()

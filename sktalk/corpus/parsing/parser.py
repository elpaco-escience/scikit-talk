import abc


class Parser(abc.ABC):
    """Abstract parser class."""

    @abc.abstractmethod
    def parse(self, file) -> "Conversation":  # noqa: F821
        return NotImplemented

import abc


class InputFile(abc.ABC):
    """Abstract parser class."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._metadata = {"source": path}

    @abc.abstractmethod
    def parse(self) -> "Conversation":
        return NotImplemented

    @property
    def metadata(self):
        metadata = self._extract_metadata()
        if metadata.keys().isdisjoint(self._metadata):
            return self._metadata | metadata
        raise ValueError("Duplicate key in the metadata")

    def _extract_metadata(self):
        return {}

    @classmethod
    def download(cls, url):              # noqa: W0613
        # download
        # downloaded_file = ...
        return NotImplemented

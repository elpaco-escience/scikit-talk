import abc


class InputFile(abc.ABC):
    """Abstract parser class."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._metadata = {"source": path}

    def parse(self) -> tuple[list["Utterance"], dict]:  # noqa: F821
        return self.utterances, self.metadata

    @property
    def metadata(self):
        metadata = self._extract_metadata()
        if metadata.keys().isdisjoint(self._metadata):
            return self._metadata | metadata
        raise ValueError("Duplicate key in the metadata")

    @property
    def utterances(self):
        self._utterances = self._extract_utterances()
        return self._utterances

    def _extract_metadata(self):
        return {}

    def _extract_utterances(self):
        return []

    @classmethod
    def download(cls, url):              # noqa: W0613
        # download
        # downloaded_file = ...
        return NotImplemented

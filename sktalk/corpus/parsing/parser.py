import abc
import datetime
from ..conversation import Conversation


class InputFile(abc.ABC):
    """Abstract parser class."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._metadata = {"source": path}

    @abc.abstractmethod
    def parse(self) -> Conversation:
        return NotImplemented

    @property
    def metadata(self):
        metadata = self._extract_metadata()
        if metadata.keys().isdisjoint(self._metadata):
            return self._metadata | metadata
        raise ValueError("Duplicate key in the metadata")

    def _extract_metadata(self):
        return {}

    @staticmethod
    def _to_timestamp(time_ms):
        try:
            time_ms = float(time_ms)
        except ValueError:
            return None
        if time_ms > 86399999:
            raise ValueError(f"timestamp {time_ms} exceeds 24h")
        if time_ms < 0:
            raise ValueError(f"timestamp {time_ms} negative")
        time_dt = datetime.datetime.utcfromtimestamp(time_ms/1000)
        return time_dt.strftime("%H:%M:%S.%f")[:-3]

    @classmethod
    def download(cls, url):              # noqa: W0613
        # download
        # downloaded_file = ...
        return NotImplemented

import abc
import datetime

from ..conversation import Conversation


class Parser(abc.ABC):
    """Abstract parser class."""

    @abc.abstractmethod
    def parse(self, file) -> Conversation:
        return NotImplemented

    @staticmethod
    def _to_timestamp(time_ms):
        try:
            time_ms = float(time_ms)
        except ValueError:
            return None
        if time_ms > 86399999:
            raise ValueError(f"timestamp {time_ms} exceeds 24h")
        elif time_ms < 0:
            raise ValueError(f"timestamp {time_ms} negative")
        time_dt = datetime.datetime.utcfromtimestamp(time_ms/1000)
        return time_dt.strftime("%H:%M:%S.%f")[:-3]

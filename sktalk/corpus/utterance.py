from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from .participant import Participant
from .select.audio import Audio


@dataclass
class Utterance(Audio):
    utterance: str
    participant: Participant = None
    time: str = None
    begin: str = None
    end: str = None
    metadata: dict[str, Any] = None

    def asdict(self):
        return asdict(self)

    # TODO function: that prints summary of data, shows it to user
    # TODO function: create a pandas data frame with the utterances

from dataclasses import dataclass, asdict
from typing import Any
from .participant import Participant


@dataclass
class Utterance:
    utterance: str
    participant: Participant = None
    time: str = None
    begin: str = None
    end: str = None
    metadata: dict[str, Any] = None

    def get_audio(self):
        pass


    # TODO function: that prints summary of data, shows it to user
    # TODO function: create a pandas data frame with the utterances

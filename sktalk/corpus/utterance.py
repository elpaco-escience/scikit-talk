from dataclasses import dataclass
from typing import Any
from .participant import Participant


@dataclass
class Utterance:
    utterance: str
    source: str
    participant: Participant = None
    time: str = None
    metadata: dict[str, Any] = None

    def get_audio(self):
        pass

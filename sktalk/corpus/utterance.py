from dataclasses import dataclass
from typing import Any
from .speaker import Speaker


@dataclass
class Utterance:
    speaker: Speaker = None
    time: str
    utterance: str
    source: str
    metadata: dict[str, Any]

    def get_audio(self):
        pass

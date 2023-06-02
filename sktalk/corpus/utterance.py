from dataclasses import dataclass
from typing import Any
from .speaker import Speaker


@dataclass
class Utterance:
    utterance: str
    source: str
    speaker: Speaker = None
    time: str = None
    metadata: dict[str, Any] = None

    def get_audio(self):
        pass

import json
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
import numpy as np
from .participant import Participant


@dataclass
class Utterance():
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


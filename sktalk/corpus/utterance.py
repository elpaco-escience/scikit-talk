from dataclasses import asdict
from dataclasses import dataclass
from math import nan
from typing import Any

from distutils.command import clean
from .participant import Participant
from collections import Counter


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

    def asdict(self):
        return asdict(self)

    # TODO function: that prints summary of data, shows it to user
    # TODO function: create a pandas data frame with the utterances

    def getnchar(self):
        clean_utt = self.utterance.replace(" ", "").strip()
        char_count = Counter(clean_utt)
        self.nchar = sum(char_count.values())
        self.length = len(clean_utt)

    def getnwords(self):
        clean_utt = self.utterance.strip()
        self.nwords = len(clean_utt.split(" "))


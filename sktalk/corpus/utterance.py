import datetime
import re
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from .participant import Participant


@dataclass
class Utterance:
    utterance: str
    participant: Participant = None
    time: list = None
    begin: str = None
    end: str = None
    metadata: dict[str, Any] = None
    utterance_clean: str = None
    utterance_list: list[str] = None
    n_words: int = None
    n_characters: int = None
    time_to_next: int = None

    def __post_init__(self):
        # clean utterance:
        # remove leading and trailing whitespace
        self.utterance_clean = self.utterance.strip()
        # remove square brackets and their contents, e.g. [laugh]
        self.utterance_clean = re.sub(r'\[[^\]]*\]', '', self.utterance_clean)
        # remove punctuation inside and outside of words
        self.utterance_clean = re.sub(r'[^\w\s]', '', self.utterance_clean)
        # remove numbers that are surrounded by spaces
        self.utterance_clean = re.sub(r'\s[0-9]+\s', ' ', self.utterance_clean)

        # generate a list of words in the utterance
        self.utterance_list = self.utterance_clean.split()

        # count words and characters
        self.n_words = len(self.utterance_list)
        self.n_characters = sum(len(word) for word in self.utterance_list)

        # calculate timestamps
        self.begin, self.end = self._split_time(self.time)

    def get_audio(self):
        pass

    def asdict(self):
        utt_dict = asdict(self)
        return utt_dict

    def until(self, next_utt):
        return next_utt.time[0] - self.time[1]

    def _split_time(self, time: list):
        if time is None:
            return None, None
        begin, end = str(time).split(", ")
        begin = begin.replace("(", "")
        end = end.replace(")", "")
        begin = self._to_timestamp(begin)
        end = self._to_timestamp(end)
        return (begin, end)

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

    # TODO function: that prints summary of data, shows it to user
    # TODO function: create a pandas data frame with the utterances

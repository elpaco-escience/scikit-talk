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
        if not self.utterance_clean:
            self._clean_utterance()

        # generate a list of words in the utterance
        if not self.utterance_list:
            self.utterance_list = self.utterance_clean.split()

        # count words and characters
        if not self.n_words:
            self.n_words = len(self.utterance_list)
        if not self.n_characters:
            self.n_characters = sum(len(word) for word in self.utterance_list)

        # calculate timestamps
        if not self.begin or not self.end:
            self._split_time()

    def get_audio(self):
        pass

    def asdict(self):
        utt_dict = asdict(self)
        return utt_dict

    def _clean_utterance(self):
        # remove leading and trailing whitespace
        self.utterance_clean = self.utterance.strip()
        # remove square brackets and their contents, e.g. [laugh]
        self.utterance_clean = re.sub(r'\[[^\]]*\]', '', self.utterance_clean)
        # remove punctuation inside and outside of words
        self.utterance_clean = re.sub(r'[^\w\s]', '', self.utterance_clean)
        # remove numbers that are surrounded by spaces
        self.utterance_clean = re.sub(r'\s[0-9]+\s', ' ', self.utterance_clean)

    def until(self, next_utt):
        return next_utt.time[0] - self.time[1]

    def _split_time(self):
        try:
            begin, end = self.time
            self.begin = self._to_timestamp(begin)
            self.end = self._to_timestamp(end)
        except (ValueError, TypeError):
            self.begin = None
            self.end = None

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

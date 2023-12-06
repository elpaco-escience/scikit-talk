import datetime
import re
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Optional


@dataclass
class Utterance:
    utterance: str
    participant: Optional[str] = None
    time: Optional[list] = None
    begin: Optional[str] = None
    end: Optional[str] = None
    utterance_clean: Optional[str] = None
    utterance_list: Optional[list[str]] = None
    n_words: Optional[int] = None
    n_characters: Optional[int] = None
    FTO: Optional[int] = None
    relevant_utterance_index: Optional[int] = None
    prior_utterance_by: Optional[str] = None
    time_to_next: Optional[int] = None
    overlap_start: Optional[int] = None
    overlapped_end: Optional[int] = None
    overlap_percentage_total: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None

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
        return asdict(self)

    def _clean_utterance(self):
        # remove leading and trailing whitespace
        self.utterance_clean = self.utterance.strip()
        # remove square brackets and their contents, e.g. [laugh]
        self.utterance_clean = re.sub(r'\[[^\]]*\]', '', self.utterance_clean)
        # remove punctuation inside and outside of words
        self.utterance_clean = re.sub(r'[^\w\s]', '', self.utterance_clean)
        # remove numbers that are surrounded by spaces
        self.utterance_clean = re.sub(r'\s\d+\s', ' ', self.utterance_clean)

    def until(self, other):
        return other.time[0] - self.time[1]

    def overlap(self, other):
        return self.window_overlap(other.time)

    def window_overlap(self, time):
        if not bool(self.time) or not bool(time):
            return None
        return self.time[1] >= time[0] and self.time[0] <= time[1]

    def overlap_duration(self, other):
        return self.window_overlap_duration(other.time)

    def window_overlap_duration(self, time):
        overlap = self.window_overlap(time)
        if not bool(overlap):
            return overlap if overlap is None else int(overlap)
        return min(self.time[1], time[1]) - max(self.time[0], time[0])

    def overlap_percentage(self, other):
        return self.window_overlap_percentage(other.time)

    def window_overlap_percentage(self, time):
        overlap_duration = self.window_overlap_duration(time)
        if not bool(overlap_duration):
            return overlap_duration
        utterance_duration = self.time[1] - self.time[0]
        return overlap_duration / utterance_duration * 100

    def same_speaker(self, other):
        return self.participant == other.participant if bool(self.participant) and bool(other.participant) else None

    def precede_with_buffer(self, other, planning_buffer=200):
        if not bool(self.time) or not bool(other.time):
            return None
        return self.time[0] - planning_buffer >= other.time[0]

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

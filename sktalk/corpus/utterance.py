import re
import warnings
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Optional


@dataclass
class Utterance:
    utterance: str
    participant: Optional[str] = None
    time: Optional[list] = None
    begin: Optional[int] = None
    begin_timestamp: Optional[str] = None
    end: Optional[int] = None
    end_timestamp: Optional[str] = None
    utterance_raw: Optional[str] = None
    utterance_list: Optional[list[str]] = None
    n_words: Optional[int] = None
    n_characters: Optional[int] = None
    FTO: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        if self.utterance_raw is None:  # if reading in existing data, we do not want to overwrite the raw utterance
            self.utterance_raw = self.utterance
        self.utterance = self._clean_utterance(self.utterance)
        self.utterance_list = self.utterance.split()
        self.n_words = len(self.utterance_list)
        self.n_characters = sum(len(word) for word in self.utterance_list)

        self._validate_time()

        if (not self.begin or not self.end) and self.time:
            self.begin = self.time[0]
            self.end = self.time[1]
            self.begin_timestamp = self._to_timestamp(self.begin)
            self.end_timestamp = self._to_timestamp(self.end)

    def get_audio(self):
        pass

    def asdict(self):
        return asdict(self)

    @classmethod
    def _fromdict(cls, fields):
        return Utterance(**fields)

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

    def _validate_time(self):
        valid = isinstance(self.time, list) and len(self.time) == 2
        valid = valid and all(isinstance(time, (float, int))
                              for time in self.time)
        valid = valid and all(time >= 0 and time < 86399999     # noqa R1716
                              for time in self.time)
        valid = valid and self.time[0] < self.time[1]

        if not valid and self.time is not None:
            warnings.warn(
                f"utterance {self.utterance} has invalid time {self.time}")
        if not valid:
            self.time = None

    @staticmethod
    def _to_timestamp(time_ms):
        time_dt = datetime.fromtimestamp(time_ms/1000, tz=timezone.utc)
        return time_dt.strftime("%H:%M:%S.%f")[:-3]

    @staticmethod
    def _clean_utterance(utterance):
        bracketed_content = r'[\[<]\w*[\]>]'  # e.g. [laugh] or <laugh>
        punctuation = r"[^\w\s']"  # except apostrophe
        numbers = r'\b\d+\b'  # only as a single word, not when inside a word
        multiple_spaces = r'\s+(?=\s{1})'

        clean_utterance = str(utterance).strip()
        for regex in [bracketed_content, punctuation, numbers, multiple_spaces]:
            clean_utterance = re.sub(regex, '', clean_utterance)

        clean_utterance = str(clean_utterance).strip()
        return clean_utterance

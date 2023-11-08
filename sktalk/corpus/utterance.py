import re
from dataclasses import asdict
from dataclasses import dataclass
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
    utterance_clean: str = None
    utterance_list: list[str] = None
    n_words: int = None
    n_characters: int = None

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

    def get_audio(self):
        pass

    def asdict(self):
        utt_dict = asdict(self)
        return utt_dict

    # TODO function: that prints summary of data, shows it to user
    # TODO function: create a pandas data frame with the utterances

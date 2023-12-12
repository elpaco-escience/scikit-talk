import re
import pylangacq
from ..utterance import Utterance
from .parser import InputFile


class ChaFile(InputFile):
    def _pla_reader(self) -> pylangacq.Reader:
        return pylangacq.read_chat(self._path)

    def _extract_metadata(self):
        return self._pla_reader().headers()[0]

    def _extract_utterances(self):
        reader = self._pla_reader().utterances(by_files=False)
        return [self._to_utterance(read_utterance) for read_utterance in reader]

    @classmethod
    def _to_utterance(cls, read_utterance) -> Utterance:
        """Convert pylangacq Utterance to sktalk Utterance"""
        participant = read_utterance.participant
        time = read_utterance.time_marks
        time = list(time) if isinstance(time, (list, tuple)) else None
        text = str(read_utterance.tiers)
        text = cls._clean_utterance(text)
        return Utterance(
            participant=participant,
            time=time,
            utterance=text,
        )

    @classmethod
    def _clean_utterance(cls, utterance):
        utterance = str(utterance)
        utterance = re.sub(r"^([^:]+):", "", utterance)
        utterance = re.sub(r"^\s+", "", utterance)
        utterance = re.sub(r"[ \t]{1,5}$", "", utterance)
        utterance = re.sub(r"\}$", "", utterance)
        utterance = re.sub(r'^\"', "", utterance)
        utterance = re.sub(r'\"$', "", utterance)
        utterance = re.sub(r"^\'", "", utterance)
        utterance = re.sub(r"\'$", "", utterance)
        utterance = re.sub(r"\\x15\d+_\d+\\x15", "", utterance)
        utterance = re.sub(r" {2}", " ", utterance)
        utterance = re.sub(r"[ \t]{1,5}$", "", utterance)
        return utterance

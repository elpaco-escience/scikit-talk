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
        with open(self._path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        utterance_info = [self._extract_info(
            line) for line in lines if not line.startswith("@")]

        # collect all utterance info in a terrible, terrible loop
        collection = []
        timing, participant, utterance = None, None, None
        for info in utterance_info:
            if info["utterance"] is not None:
                timing = info["time"]
                utterance = info["utterance"]
            else:
                continue
            if info["participant"] is not None:
                participant = info["participant"]
            complete_utterance = Utterance(
                participant=participant,
                time=timing,
                utterance=utterance)
            collection.append(complete_utterance)
        return collection

    @classmethod
    def _extract_info(cls, line):
        participant = cls._extract_participant(line)
        time = cls._extract_timing(line)
        utterance = cls._extract_utterance(line)
        return ({"participant": participant,
                "time": time,
                 "utterance": utterance})

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

    @classmethod
    def _extract_timing(cls, line):
        timing = re.search(r"[0-9]{1,9}_[0-9]{1,9}", line)
        try:
            timing = timing.group()
            timing = timing.split("_")
            timing = [int(t) for t in timing]
        except AttributeError:
            timing = None
        return timing

    @classmethod
    def _extract_participant(cls, line):
        part_re = re.search(r"(?<=\*)\S+(?=\:)", line)
        try:
            participant = part_re.group()
        except AttributeError:
            participant = None
        return participant

    @classmethod
    def _extract_utterance(cls, line):
        utt_re = re.search(r"(?<=\t).*(?=\s.{1}[0-9]{1,9}_[0-9]{1,9})", line)
        try:
            utterance = utt_re.group()
            if re.match(r"\([0-9.]+\)", utterance):
                utterance = None
        except AttributeError:
            utterance = None

        return utterance

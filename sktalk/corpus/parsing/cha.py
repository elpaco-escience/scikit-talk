import re
import pylangacq
from ..utterance import Utterance
from .parser import InputFile


class ChaFile(InputFile):
    PARTICIPANT_REGEX = r"^\*(?P<participant>\S+)\:"
    TIMING_REGEX = r"(?P<timing>\d{1,9}_\d{1,9})"
    UTTERANCE_REGEX = r"\t(?P<utterance>.*)\s." + TIMING_REGEX
    SPACER_REGEX = r"\([0-9.]+\)"

    def _pla_reader(self) -> pylangacq.Reader:
        return pylangacq.read_chat(self._path)

    def _extract_metadata(self):
        return self._pla_reader().headers()[0]

    def _extract_utterances(self):
        with open(self._path, encoding="utf-8") as f:
            lines = f.readlines()
        utterance_info = [self._extract_info(
            line) for line in lines if not line.startswith("@")]

        # collect all utterance info in a terrible, terrible loop
        collection = []
        timing, participant, utterance = None, None, None
        for info in utterance_info:
            if info["utterance"] is None:
                continue
            if info["utterance"] is not None:
                timing = info["time"]
                utterance = info["utterance"]
            if info["participant"] is not None:
                participant = info["participant"]
            complete_utterance = Utterance(
                participant=participant,
                time=timing,
                utterance=utterance)
            collection.append(complete_utterance)
        return collection

    @staticmethod
    def _extract_info(line):
        participant = ChaFile._extract_participant(line)
        time = ChaFile._extract_timing(line)
        utterance = ChaFile._extract_utterance(line)
        return ({"participant": participant,
                "time": time,
                 "utterance": utterance})

    @staticmethod
    def _clean_utterance(utterance):
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

    @staticmethod
    def _extract_timing(line):
        timing = re.search(ChaFile.TIMING_REGEX, line)
        try:
            timing = timing.group("timing")
            timing = timing.split("_")
            timing = [int(t) for t in timing]
        except AttributeError:
            timing = None
        return timing

    @staticmethod
    def _extract_participant(line):
        part_re = re.search(ChaFile.PARTICIPANT_REGEX, line)
        try:
            participant = part_re.group("participant")
        except AttributeError:
            participant = None
        return participant

    @staticmethod
    def _extract_utterance(line):
        utt_re = re.search(ChaFile.UTTERANCE_REGEX, line)
        try:
            utterance = utt_re.group("utterance")
            if re.match(ChaFile.SPACER_REGEX, utterance):
                utterance = None
        except AttributeError:
            utterance = None
        return utterance

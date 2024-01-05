import re
import pylangacq
from ..utterance import Utterance
from .parser import InputFile


class ChaFile(InputFile):
    # participant may not be repeated on each line, hence the ?
    PARTICIPANT_REGEX = r"(^\*(?P<participant>[^:]+)\:)?"
    UTTERANCE_REGEX = r"\s+(?P<utterance>.*)\s+"
    TIMING_REGEX = r"\D(?P<timing>\d{1,9}_\d{1,9})\D"
    LINE_REGEX = PARTICIPANT_REGEX + UTTERANCE_REGEX + TIMING_REGEX

    SPACER_REGEX = r"\((?P<spacer>[\d.]+)\)"

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
        participant = None
        for info in utterance_info:
            if info["utterance"] is None:
                continue
            if info["participant"] is not None:
                participant = info["participant"]
            complete_utterance = Utterance(
                participant=participant,
                time=info["time"],
                utterance=info["utterance"])
            collection.append(complete_utterance)
        return collection

    @staticmethod
    def _extract_info(line):
        default_return = {"utterance": None}

        extract_re = re.search(ChaFile.LINE_REGEX, line)
        if not extract_re:
            return default_return

        try:
            utterance = ChaFile._clean_utterance(extract_re["utterance"])
        except TypeError:
            return default_return

        if utterance is not None:
            timing = ChaFile._clean_timing(extract_re["timing"])
            return ({
                "participant": extract_re["participant"],
                "time": timing,
                "utterance": utterance
            })
        return default_return

    @staticmethod
    def _clean_utterance(utterance):
        if re.match(ChaFile.SPACER_REGEX, utterance):
            return None
        return str(utterance).strip()

    @staticmethod
    def _clean_timing(timing):
        timing = timing.split("_")
        return [int(t) for t in timing] if len(timing) == 2 else None

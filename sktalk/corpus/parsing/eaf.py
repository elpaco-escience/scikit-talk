from ..utterance import Utterance
from .parser import InputFile


class EafFile(InputFile):

    def _extract_metadata(self):
        return {}

    def _extract_utterances(self):
        return []

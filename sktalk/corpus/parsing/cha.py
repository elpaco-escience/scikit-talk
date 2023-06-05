import pylangacq

from ..conversation import Conversation
from ..utterance import Utterance
from .parser import Parser


class ChaParser(Parser):
    def parse(self, file):
        chatfile = pylangacq.read_chat(file)
        chat_utterances = chatfile.utterances(by_files=False)

        utterances = [ChaParser._to_utterance(row, file) for row in chat_utterances]
        return Conversation(utterances)

    @staticmethod
    def _to_utterance(row, file):
        return Utterance(
            speaker=row.participant,
            time=str(row.time_marks),
            utterance=str(row.tiers),
            source=file,
        )

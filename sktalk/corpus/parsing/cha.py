from .parser import Parser
from ..conversation import Conversation
from ..utterance import Utterance

import pylangacq


class ChaParser(Parser):
    def parse(self, file):
        chatfile = pylangacq.read_chat(file)
        chat_utterances = chatfile.utterances(by_files=False)

        utterances = [self._to_utterance(row, file) for row in chat_utterances]
        return Conversation(utterances)

    def _to_utterance(self, row, file):
        return Utterance(
            speaker=row.participant,
            time=str(row.time_marks),
            utterance=str(row.tiers),
            source=file,
        )

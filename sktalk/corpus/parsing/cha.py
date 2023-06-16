import pylangacq
import re

from ..conversation import Conversation
from ..utterance import Utterance
from .parser import Parser


class ChaParser(Parser):
    def parse(self, file):
        chatfile = pylangacq.read_chat(file)
        chat_utterances = chatfile.utterances(by_files=False)

        utterances = [ChaParser._to_utterance(
            row, file) for row in chat_utterances]
        return Conversation(utterances)

    @staticmethod
    def _to_utterance(row, file):
        utterance = Utterance(
            participant=row.participant,
            time=row.time_marks,
            utterance=str(row.tiers),
            source=file,
        )
        utterance.begin, utterance.end = ChaParser._split_time(utterance.time)
        return utterance

    @staticmethod
    def _split_time(time):
        if time is None:
            return None, None
        time = str(time).split(', ')
        begin, end = time[0], time[1]
        begin = re.sub(r'\(', "", begin)
        end = re.sub(r'\)', "", end)
        begin = Parser._to_timestamp(begin)
        end = Parser._to_timestamp(end)
        return (begin, end)

import re
import pylangacq
from ..conversation import Conversation
from ..utterance import Utterance
from .parser import InputFile


class ChaFile(InputFile):
    def _pla_reader(self) -> pylangacq.Reader:
        return pylangacq.read_chat(self._path)
    
    def parse(self):
        """Parse conversation file in Chat format

        Returns:
            Conversation: A Conversation object representing the conversation in the file.
        """
        chat_utterances = self._pla_reader().utterances(by_files=False)

        utterances = [ChaFile._to_utterance(
            chat_utterance) for chat_utterance in chat_utterances]

        return Conversation(utterances, self.metadata)

    @staticmethod
    def _to_utterance(chat_utterance) -> Utterance:
        utterance = Utterance(
            participant=chat_utterance.participant,
            time=chat_utterance.time_marks,
            utterance=str(chat_utterance.tiers),
        )
        utterance.begin, utterance.end = ChaFile._split_time(utterance.time)
        utterance.utterance = ChaFile._clean_utterance(utterance.utterance)
        return utterance

    def _extract_metadata(self):
        return self._pla_reader().headers()[0]


    @staticmethod
    def _split_time(time):
        if time is None:
            return None, None
        begin, end = str(time).split(", ")
        begin = begin.replace("(", "")
        end = end.replace(")", "")
        begin = InputFile._to_timestamp(begin)
        end = InputFile._to_timestamp(end)
        return (begin, end)

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

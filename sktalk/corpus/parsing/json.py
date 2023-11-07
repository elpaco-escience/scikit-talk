from .parser import InputFile
from ..conversation import Conversation
from ..utterance import Utterance
import json


class JsonFile(InputFile):
    def parse(self) -> "Conversation":  # noqa: F821
        """Parse conversation file in JSON format

        Returns:
            Conversation: A Conversation object representing the conversation in the file.
        """
        with open(self._path) as f:
            json_in = json.load(f)
        utterances = [Utterance("")]
        metadata = json_in
        return Conversation(utterances, metadata)

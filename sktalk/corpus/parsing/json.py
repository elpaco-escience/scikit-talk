import json
from ..conversation import Conversation
from ..utterance import Utterance
from .parser import InputFile


class JsonFile(InputFile):
    def parse(self) -> "Conversation":  # noqa: F821
        """Parse conversation file in JSON format

        Returns:
            Conversation: A Conversation object representing the conversation in the file.
        """
        with open(self._path, encoding='utf-8') as f:
            json_in = json.load(f)
        utterances = [Utterance(**u) for u in json_in["Utterances"]]
        del json_in["Utterances"]
        return Conversation(utterances, metadata=json_in)

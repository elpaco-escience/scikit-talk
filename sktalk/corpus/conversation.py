import json
import warnings
from .utterance import Utterance
from .write.writer import Writer


class Conversation(Writer):
    def __init__(
        self, utterances: list["Utterance"], metadata: dict = None  # noqa: F821
    ) -> None:
        """Representation of a transcribed conversation

        Args:
            utterances (list[Utterance]): A list of Utterance objects representing the utterances in the conversation.
            metadata (dict, optional): Additional metadata associated with the conversation. Defaults to None.
        """
        self._metadata = metadata or {}
        self._utterances = utterances
        # Input utterances should be a list of type Utterance
        errormsg = "All utterances in a conversation should be of type Utterance"
        if not isinstance(self._utterances, list):
            try:
                self._utterances = list(self._utterances)
            except TypeError as e:
                raise TypeError(errormsg) from e
        for utterance in self._utterances:
            if not isinstance(utterance, Utterance):
                raise TypeError(errormsg)
        # The list can be empty. This would be weird and the user needs to be warned.
        if not self._utterances:
            warnings.warn(
                "This conversation appears to be empty: no Utterances are read.")

    @property
    def utterances(self):
        """
        Get the list of utterances in the conversation.

        Returns:
            list[Utterance]: A list of Utterance objects representing the utterances in the conversation.
        """
        return self._utterances

    @property
    def metadata(self):
        """
        Get the metadata associated with the conversation.

        Returns:
            dict: Additional metadata associated with the conversation.
        """
        return self._metadata

    @classmethod
    def from_json(cls, path):
        """Parse conversation file in JSON format

        Returns:
            Conversation: A Conversation object representing the conversation in the file.
        """
        with open(path, encoding='utf-8') as f:
            json_in = json.load(f)      
        utterances = [Utterance.fromdict(u) for u in json_in["Utterances"]]
        del json_in["Utterances"]
        return Conversation(utterances, metadata=json_in)

    def get_utterance(self, index) -> "Utterance":  # noqa: F821
        raise NotImplementedError

    def summarize(self):
        """
        Print a summary of the conversation.
        """
        for utterance in self._utterances[:10]:
            print(utterance)

    def asdict(self):
        """
        Return the Conversation as a dictionary

        Returns:
            dict: dictionary containing Conversation metadata and Utterances
        """
        return self._metadata | {"Utterances": [u.asdict() for u in self._utterances]}

import warnings
from .utterance import Utterance
from .write.writer import Writer
from pathlib import Path

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

    def write_csv(self, path: str = "./file.csv"):
        _path = Path(path).with_suffix(".csv")
        path_metadata = self._specify_path(_path,"metadata")
        path_participants = self._specify_path(_path,"participants")
        path_utterances = self._specify_path(_path,"utterances")
        self._write_csv_metadata(path_metadata)
        self._write_csv_participants(path_participants)
        self._write_csv_utterances(path_utterances)


    def _write_csv_metadata(self, path: str):
        headers = self._metadata.keys()
        self._write_csv(path, headers, [self._metadata])

    def _write_csv_utterances(self, path: str):
        rows = [utterance.asdict() for utterance in self._utterances]
        headers = rows[0].keys()
        self._write_csv(path, headers, rows)

    def _write_csv_participants(self, path: str):
        return NotImplemented

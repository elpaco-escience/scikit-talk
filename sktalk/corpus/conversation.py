import os
import json

class Conversation:
    def __init__(
        self, utterances: list["Utterance"], metadata: dict = None  # noqa: F821
    ) -> None:
        """Representation of a transcribed conversation

        Args:
            utterances (list[Utterance]): A list of Utterance objects representing the utterances in the conversation.
            metadata (dict, optional): Additional metadata associated with the conversation. Defaults to None.
        """
        self._utterances = utterances
        self._metadata = metadata

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

    def to_json(self, name, directory):
        """
        Dump a conversation to a JSONL file.

        Args:
            name (str): The name of the corpus that will be the file name.
            directory (str): The path to the directory where the .jsonl
            file will be saved.
        """
        name = f"{name}.jsonl"
        destination = os.path.join(directory, name)

        utt_list = [u.to_dict() for u in self._utterances]
        conv_dict = self._metadata.copy()
        conv_dict["Conversation"] = utt_list

        with open(destination, "w") as file:
            json.dump(conv_dict, file, indent=4)

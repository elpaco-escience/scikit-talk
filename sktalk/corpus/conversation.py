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

    def subconversation(self,
                        index: int,
                        before: int = 0,
                        after: int = 0,
                        time_or_index: str = "index"):
        # select utterance based on the value of utterance.begin
        # type index = select utterance based on index
        # type time = select utterance based on time

        # verify if index is within range:
        if index < 0 or index >= len(self.utterances):
            raise IndexError("Index out of range")
        if time_or_index == "time":
            begin = self.utterances[index].time[0] - before
            end = self.utterances[index].time[1] + after
            [u for u in self.utterances if u.time[0] >= begin and u.time[1] <= end]
        elif time_or_index == "index":
            # check if selection is within range
            if index - before < 0 or index + after + 1 > len(self.utterances):
                raise IndexError("Index out of range")
        else:
            raise ValueError("time_or_index must be either 'time' or 'index'")

        # start with utterance
        # obtain utterance context; search criteria may be time, or [i]
        # create a new conversation object from this
        return Conversation(self.utterances[index-before:index+after+1], self.metadata)

    @property
    def until_next(self):
        if len(self.utterances) != 2:
            raise ValueError("Conversation must have 2 utterances")
        return self.utterances[0].until(self.utterances[1])

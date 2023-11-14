import warnings
from typing import Optional
from .utterance import Utterance
from .write.writer import Writer


class Conversation(Writer):
    def __init__(
        self, utterances: list["Utterance"], metadata: Optional[dict] = None  # noqa: F821
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
                        after: Optional[int] = None,
                        time_or_index: str = "index") -> "Conversation":
        """Select utterances to provide context as a sub-conversation

        Args:
            index (int): The index of the utterance for which to provide context
            before (int, optional): Either the number of utterances prior to indicated utterance,
                                or the time in ms preceding the utterance's begin. Defaults to 0.
            after (int, optional): Either the number of utterances after the indicated utterance,
                                or the time in ms following the utterance's end. Defaults to None,
                                which then assumes the same value as `before`.
            time_or_index (str, optional): Use "time" to select based on time (in ms), or "index"
                                to select a set number of utterances irrespective of timing.
                                Defaults to "index".

        Raises:
            IndexError: Index provided must be within range of utterances
            ValueError: time_or_index must be either "time" or "index"

        Returns:
            Conversation: Conversation object containing a reduced set of utterances
        """
        if index < 0 or index >= len(self.utterances):
            raise IndexError("Index out of range")
        if after is None:
            after = before
        if time_or_index == "index":
            # if before/after would exceed the bounds of the list, adjust
            if index - before < 0:
                before = index
            if index + after + 1 > len(self.utterances):
                after = len(self.utterances) - index - 1
            returned_utterances = self.utterances[index-before:index+after+1]
        elif time_or_index == "time":
            begin = self.utterances[index].time[0] - before
            end = self.utterances[index].time[1] + after
            returned_utterances = [
                u for u in self.utterances if self.overlap(begin, end, u.time)]
        else:
            raise ValueError("time_or_index must be either 'time' or 'index'")

        return Conversation(returned_utterances, self.metadata)

    @property
    def until_next(self):
        if len(self.utterances) != 2:
            raise ValueError("Conversation must have 2 utterances")
        return self.utterances[0].until(self.utterances[1])

    @property
    def dyadic(self) -> bool:
        participants = [u.participant for u in self.utterances]
        return len(set(participants)) == 2

    @staticmethod
    def overlap(begin: int, end: int, time: list):
        # there is overlap if:
        # time[0] falls between begin and end
        # time[1] falls between and end
        # time[0] is before begin and time[1] is after end
        if begin <= time[0] <= end or begin <= time[1] <= end:
            return True
        return time[0] <= begin and time[1] >= end

import warnings
from typing import Optional
from .utterance import Utterance
from .write.writer import Writer


class Conversation(Writer):
    def __init__(
        self, utterances: list["Utterance"], metadata: Optional[dict] = None, suppress_warnings: bool = False  # noqa: F821
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
        if not self._utterances and not suppress_warnings:
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

    def _subconversation(self,
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
        # TODO consider adding parameter 'strict' that only returns utterances entirely inside the window
        if index < 0 or index >= len(self._utterances):
            raise IndexError("Index out of range")
        if after is None:
            after = before
        if time_or_index == "index":
            # if before/after would exceed the bounds of the list, adjust
            if index - before < 0:
                before = index
            if index + after + 1 > len(self._utterances):
                after = len(self._utterances) - index - 1
            returned_utterances = self._utterances[index-before:index+after+1]
        elif time_or_index == "time":
            try:
                begin = self._utterances[index].time[0] - before
                end = self._utterances[index].time[1] + after
                returned_utterances = [
                    u for u in self._utterances if self.overlap(begin, end, u.time)]
            except (TypeError, IndexError):
                return Conversation([], suppress_warnings=True)
        else:
            raise ValueError(
                "`time_or_index` must be either 'time' or 'index'")
        return Conversation(utterances=returned_utterances)

    def count_participants(self) -> int:
        """Count the number of participants in a conversation

        Importantly: if one of the utterances has no participant, it is counted
        as a separate participant (None).

        Returns:
            int: number of participants
        """
        participants = [u.participant for u in self.utterances]
        return len(set(participants))

    def _update(self, field: str, values: list, **kwargs):
        """
        Update the all utterances in the conversation with calculated values

        This function also stores relevant arguments in the Conversation metadata.

        Args:
            field (str): field of the Utterance to update
            values (list): list of values to update each utterance with
            kwargs (dict): information about the calculation to store in the Conversation metadata
        """
        if len(values) != len(self.utterances):
            raise ValueError(
                "The number of values must match the number of utterances")
        metadata = {field: kwargs}
        try:
            self._metadata["Calculations"].update(metadata)
        except KeyError:
            self._metadata = {"Calculations": metadata}
        for index, utterance in enumerate(self.utterances):
            setattr(utterance, field, values[index])

    def calculate_FTO(self, window: int = 10000, planning_buffer: int = 200, n_participants: int = 2):
        """Calculate Floor Transfer Offset (FTO) per utterance

        FTO is defined as the difference between the time that a turn starts and the
        end of the most relevant prior turn by the other participant, which is not
        necessarily the prior utterance.

        An utterance does not receive an FTO if there are preceding utterances
        within the window that do not have timing information, or if it lacks
        timing information itself.

        To be a relevant prior turn, the following conditions must be met, respective to utterance U:
        - the utterance must be by another speaker than U
        - the utterance by the other speaker must be the most recent utterance by that speaker
        - the utterance must have started before utterance U, more than `planning_buffer` ms before.
        - the utterance must be partly or entirely within the context window (`window` ms prior to the start of utterance U)
        - within the context window, there must be a maximum of `n_participants` speakers.

        Args:
            window (int, optional): _description_. Defaults to 10000.
            planning_buffer (int, optional): _description_. Defaults to 200.
            n_participants (int, optional): _description_. Defaults to 2.
        """
        values = []
        for index, utterance in enumerate(self.utterances):
            sub = self._subconversation(
                index=index,
                time_or_index="time",
                before=window,
                after=0)
            if not 2 <= sub.count_participants() <= n_participants:
                values.append(None)
                continue
            potentials = [
                u for u in sub.utterances if utterance.relevant_for_fto(u, planning_buffer)]
            try:
                relevant = potentials[-1]
                values.append(relevant.until(utterance))
            except IndexError:
                values.append(None)
        self._update("FTO", values,
                     window=window,
                     planning_buffer=planning_buffer,
                     n_participants=n_participants)

    @staticmethod
    def overlap(begin: int, end: int, time: list):
        # there is overlap if:
        # time[0] falls between begin and end
        # time[1] falls between and end
        # time[0] is before begin and time[1] is after end
        if time is None:
            return False
        if begin <= time[0] <= end or begin <= time[1] <= end:
            return True
        return time[0] <= begin and time[1] >= end

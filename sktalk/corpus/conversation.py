import json
import warnings
from typing import Optional
import pandas as pd
from .parsing.cha import ChaFile
from .parsing.eaf import EafFile
from .utterance import Utterance
from .write.writer import Writer


class Conversation(Writer):
    def __init__(
        self,
        utterances: list["Utterance"],
        metadata: Optional[dict] = None,
        suppress_warnings: bool = False
    ) -> None:
        """Representation of a transcribed conversation

        Args:
            utterances (list[Utterance]): A list of Utterance objects representing the utterances in the conversation.
            metadata (dict, optional): Additional metadata associated with the conversation. Defaults to None.
        """
        self._metadata = metadata or {"source": "unknown"}

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

        self._metadata_df = None
        self._utterance_df = None

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

    def __len__(self):
        """
        Get the number of utterances in the conversation.

        Returns:
            int: The number of utterances in the conversation.
        """
        return len(self._utterances)

    @property
    def participants(self):
        """
        Get the participants in the conversation.

        Returns:
            set[str]: A set of unique participant names.
        """
        return {u.participant for u in self._utterances}

    @classmethod
    def from_cha(cls, path):
        """Parse conversation file in Cha format

        Args:
            path (str): Path to the Cha file

        Returns:
            Conversation: A Conversation object representing the conversation in the file.
        """
        utterances, metadata = ChaFile(path).parse()
        return cls(utterances, metadata)

    @classmethod
    def from_eaf(cls, path: str, tiers: Optional[list[str]] = None):
        """Parse conversation file in ELAN format

        Args:
            path (str): Path to the ELAN file
            tiers (Optional[list[str]], optional): List of tiers to parse. Defaults to None, in which case all tiers are parsed.
                If an empty list is passed, all tiers are parsed, but a warning is issued.

        Raises:
            KeyError if tiers are named that are not found in the file.

        Returns:
            Conversation: A Conversation object representing the conversation in the file.
        """
        utterances, metadata = EafFile(path, tiers).parse()
        return cls(utterances, metadata)

    @classmethod
    def from_json(cls, path):
        """Parse conversation file in JSON format

        Returns:
            Conversation: A Conversation object representing the conversation in the file.
        """
        with open(path, encoding='utf-8') as f:
            json_in = json.load(f)
        return cls._fromdict(json_in)

    @classmethod
    def _fromdict(cls, fields):
        try:
            utterances = [Utterance._fromdict(u) for u in fields["Utterances"]]
            del fields["Utterances"]
        except KeyError as e:
            raise TypeError(
                "This object cannot be imported as a Conversation.") from e
        return Conversation(utterances, metadata=fields)

    def get_utterance(self, index) -> "Utterance":  # noqa: F821
        raise NotImplementedError

    def summary(self, n=10, **fields):
        """
        Print the first n lines of a conversation.

        Args:
            n (int, optional): Number of lines to print. Defaults to 10.
            fields (dict): key-value pairs with which specific utterances can be selected
        """
        selected = self.select(**fields)
        for u in selected.utterances[:n]:
            if len(u.time) != 2:
                time = "(no timing information)"
            else:
                time = f"({u.time[0]} - {u.time[1]})"
            print(f"{time} {u.participant}: '{u.utterance}'")

    def select(self, **fields):
        """Select utterances based on content in specific fields

        Args:
            fields (dict): key-value pairs with which specific utterances can be selected

        Returns:
            Conversation: Conversation object without metadata, containing a reduced set of utterances
        """
        utterances = [utterance for utterance in self._utterances if all(
            getattr(utterance, key) == value for key, value in fields.items())]
        return Conversation(utterances, suppress_warnings=True)

    def remove(self, **fields):
        """Remove utterances based on content in specific fields

        Args:
            fields (dict): key-value pairs with which specific utterances can be selected
        """
        to_remove = self.select(**fields)
        self._utterances = [
            u for u in self._utterances if u not in to_remove.utterances]

    def asdict(self):
        """
        Return the Conversation as a dictionary

        Returns:
            dict: dictionary containing Conversation metadata and Utterances
        """
        return self._metadata | {"Utterances": [u.asdict() for u in self._utterances]}

    @property
    def metadata_df(self):
        """Return the conversation metadata as a pandas dataframe."""
        if self._metadata_df is None:
            self._metadata_df = self._metadata_to_df(self._metadata)
        return self._metadata_df

    @property
    def utterance_df(self):
        """Return the conversation utterances as a pandas dataframe."""
        if self._utterance_df is None:
            self._utterance_df = pd.DataFrame(self._utterances)
            self._utterance_df.insert(loc=0,
                                      column="source",
                                      value=self._metadata["source"])
        return self._utterance_df

    def _subconversation_by_index(self,
                                  index: int,
                                  before: int = 0,
                                  after: Optional[int] = None) -> "Conversation":
        """Select utterances to provide context as a sub-conversation

        Args:
            index (int): The index of the utterance for which to provide context
            before (int, optional): The number of utterances prior to indicated utterance. Defaults to 0.
            after (int, optional): The number of utterances after the indicated utterance. Defaults to None,
                which then assumes the same value as `before`.

        Raises:
            IndexError: Index provided must be within range of utterances

        Returns:
            Conversation: Conversation object without metadata, containing a reduced set of utterances
        """
        if index < 0 or index >= len(self._utterances):
            raise IndexError("Utterance index out of range")
        if after is None:
            after = before
        left_bound = max(index-before, 0)
        right_bound = min(index + after + 1, len(self._utterances))
        return Conversation(utterances=self._utterances[left_bound:right_bound],
                            suppress_warnings=True)

    def _subconversation_by_time(self,
                                 index: int,
                                 before: int = 0,
                                 after: int = 0,
                                 exclude_utterance_overlap: bool = False) -> "Conversation":
        """Select utterances to provide context as a sub-conversation

        Args:
            index (int): The index of the utterance for which to provide context
            before (int, optional): The time in ms preceding the utterance's begin. Defaults to 0.
            after (int, optional): The time in ms following the utterance's end. Defaults to 0
            exclude_utterance_overlap (bool, optional): If True, the duration of the
                utterance itself is not used to identify overlapping utterances, and only
                the window before or after the utterance is used. Defaults to False.
                If True, only one of `before` or `after` can be more than 0, as the window
                for overlap will be limited to the window preceding or following the utterance.

        Returns:
            Conversation: Conversation object without metadata, containing a reduced set of utterances
        """
        if index < 0 or index >= len(self._utterances):
            raise IndexError("Utterance index out of range")
        if exclude_utterance_overlap and before > 0 and after > 0:
            raise ValueError(
                "When utterance is excluded from overlap window, only one of before or after can be more than 0")
        try:
            begin = self._utterances[index].time[0] - before
            end = self._utterances[index].time[1] + after
            left_bound, right_bound = None, None
            if exclude_utterance_overlap and before == 0:  # only overlap with window following utterance
                begin = self._utterances[index].time[1]
                left_bound = index
            elif exclude_utterance_overlap and after == 0:  # only overlap with window preceding utterance
                end = self._utterances[index].time[0]
                right_bound = index + 1
            indices = [i for i, u in enumerate(
                self._utterances) if u.window_overlap([begin, end])]
            left_bound = left_bound if bool(left_bound) else min(indices)
            right_bound = right_bound if bool(
                right_bound) else max(indices) + 1
            returned_utterances = self._utterances[left_bound:right_bound]
        except (TypeError, IndexError):
            # if the utterance's timing is None, a TypeError is raised
            # if the utterance has no time[0] or time[1], an IndexError is raised
            # In both cases, there is missing timing information, so no data can be returned.
            returned_utterances = []
        return Conversation(utterances=returned_utterances, suppress_warnings=True)

    def count_participants(self, except_none: bool = False) -> int:
        """Count the number of participants in a conversation

        Importantly: if one of the utterances has no participant, it is counted
        as a separate participant (None). If you want to exclude these, set
        `except_none` to True.

        Args:
            except_none (bool, optional): if `True`, utterances without a participant are not counted. Defaults to `False`.

        Returns:
            int: number of participants
        """
        participants = self.participants
        if except_none:
            participants = [p for p in participants if p is not None]
        return len(participants)

    def _update(self, field: str, values: list, **kwargs):
        """
        Update all utterances in the conversation with calculated values

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
            self._metadata = self._metadata | {"Calculations": metadata}
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

        Args:
            window (int, optional): the time in ms prior to utterance in which a
                relevant preceding utterance can be found. Defaults to 10000.
            planning_buffer (int, optional): minimum speaking time in ms to allow for a response.
                Defaults to 200.
            n_participants (int, optional): maximum number of participants overlapping with
                the utterance and preceding window. Defaults to 2.
        """
        values = []
        for index, utterance in enumerate(self.utterances):
            relevant = self.relevant_prior_utterance(
                index, window, planning_buffer, n_participants)
            values.append(relevant.until(utterance)
                          if bool(relevant) else None)
        self._update("FTO", values,
                     window=window,
                     planning_buffer=planning_buffer,
                     n_participants=n_participants)

    def relevant_prior_utterance(self,
                                 index,
                                 window=10000,
                                 planning_buffer=200,
                                 n_participants=2):
        """Determine the most relevant prior utterance for a given utterance

        To be a relevant prior turn, the following conditions must be met, respective to utterance U:
        - the utterance must be by another speaker than U
        - the utterance by the other speaker must be the most recent utterance by that speaker
        - the utterance must have started before utterance U, more than `planning_buffer` ms before.
        - the utterance must be partly or entirely within the context window (`window` ms prior
            to the start of utterance U)
        - within the context window, there must be a maximum of `n_participants` speakers.

        Args:
            index (int): index of the utterance to assess
            window (int, optional): the time in ms prior to utterance in which a
                relevant preceding utterance can be found. Defaults to 10000.
            planning_buffer (int, optional): minimum speaking time in ms to allow for a response.
                Defaults to 200.
            n_participants (int, optional): maximum number of participants overlapping with
                the utterance and preceding window. Defaults to 2.

        Returns:
            Utterance: the most relevant prior utterance, or None, if no relevant prior utterance can be identified
        """
        utterance_u = self._utterances[index]
        if not bool(utterance_u.time) or not bool(utterance_u.participant):
            return None
        sub = self._subconversation_by_time(
            index=index,
            before=window,
            after=0,
            exclude_utterance_overlap=True)
        if not 2 <= sub.count_participants() <= n_participants:
            return None
        must_overlap = []
        for prior in sub.utterances[::-1]:
            # if timing or participant information is missing, stop looking for relevant utterances
            if not bool(prior.time) or not bool(prior.participant):
                break
            if prior == utterance_u:
                continue
            # if the utterance is by the same speaker, it is not relevant,
            # but must overlap with potential relevant utterance
            if utterance_u.same_speaker(prior):
                must_overlap.append(prior)
                continue
            # the relevant utterance must precede utterance U more than planning buffer
            if not utterance_u.precede_with_buffer(prior, planning_buffer):
                continue
            # verify that all utterances in must_overlap do so
            if all(utt.overlap_percentage(prior) == 100 for utt in must_overlap):
                return prior
        return None

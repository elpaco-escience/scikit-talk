"""Documentation about the scikit-talk module - csv to json."""
import json
import pandas as pd
from pandas import DataFrame
# import random
# import shutil
# from typing import Collection, Callable, Set, Generator, Tuple, ValuesView, Union
from typing import Optional, List, Dict, Union, Iterable, Callable, Generator
# from pandas import DataFrame
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from collections import defaultdict, deque
import os
from yaml import load, Loader

try:
    from collections.abc import MutableMapping
except:
    from collections import MutableMapping

# from convokit.convokitConfig import ConvoKitConfig
# from convokit.util import create_safe_id
# from .convoKitMatrix import ConvoKitMatrix
# from .corpusUtil import *
# from .corpus_helpers import *
# from .storageManager import StorageManager


### CLASSES

class CorpusComponent:
    def __init__(
        self,
        obj_type: str,
        owner=None,
        id=None,
        initial_data=None,
        vectors: List[str] = None,
        meta=None,
    ):
        self.obj_type = obj_type  # utterance, speaker, conversation
        self._owner = owner
        self._id = id
        self.vectors = vectors if vectors is not None else []

        # if the CorpusComponent is initialized with an owner set up an entry
        # in the owner's storage; if it is not initialized with an owner
        # (i.e. it is a standalone object) set up a dict-based temp storage
        if self.owner is None:
            self._temp_storage = initial_data if initial_data is not None else {}
        else:
            self.owner.storage.initialize_data_for_component(
                self.obj_type,
                self._id,
                initial_value=(initial_data if initial_data is not None else {}),
            )

        if meta is None:
            meta = dict()
        self._meta = self.init_meta(meta)

    def get_owner(self):
        return self._owner

    def set_owner(self, owner):
        if owner is self._owner:
            # no action needed
            return
        # stash the metadata first since reassigning self._owner will break its storage connection
        meta_vals = {k: v for k, v in self.meta.items()}
        previous_owner = self._owner
        self._owner = owner
        if owner is not None:
            # when a new owner Corpus is assigned, we must take the following steps:
            # (1) transfer this component's data to the new owner's StorageManager
            # (2) avoid duplicates by removing the data from the old owner (or temp storage if there was no prior owner)
            # (3) reinitialize the metadata instance
            data_dict = (
                dict(previous_owner.storage.get_data(self.obj_type, self.id))
                if previous_owner is not None
                else self._temp_storage
            )
            self.owner.storage.initialize_data_for_component(
                self.obj_type, self.id, initial_value=data_dict
            )
            if previous_owner is not None:
                previous_owner.storage.delete_data(self.obj_type, self.id)
                previous_owner.storage.delete_data("meta", self.meta.storage_key)
            else:
                del self._temp_storage
            self._meta = self.init_meta(meta_vals)

    owner = property(get_owner, set_owner)

    def init_meta(self, meta, overwrite=False):
        if self._owner is None:
            # ConvoKitMeta instances are not allowed for ownerless (standalone)
            # components since they must be backed by a StorageManager. In this
            # case we must forcibly convert the ConvoKitMeta instance to dict
            if isinstance(meta, ConvoKitMeta):
                meta = meta.to_dict()
            return meta
        else:
            if isinstance(meta, ConvoKitMeta) and meta.owner is self._owner:
                return meta
            ck_meta = ConvoKitMeta(self, self.owner.meta_index, self.obj_type, overwrite=overwrite)
            for key, value in meta.items():
                ck_meta[key] = value
            return ck_meta

    def get_id(self):
        return self._id

    def set_id(self, value):
        if not isinstance(value, str) and value is not None:
            self._id = str(value)
            warn(
                "{} id must be a string. ID input has been casted to a string.".format(
                    self.obj_type
                )
            )
        else:
            self._id = value

    id = property(get_id, set_id)

    def get_meta(self):
        return self._meta

    def set_meta(self, new_meta):
        self._meta = self.init_meta(new_meta, overwrite=True)

    meta = property(get_meta, set_meta)

    def get_data(self, property_name):
        if self._owner is None:
            return self._temp_storage[property_name]
        return self.owner.storage.get_data(self.obj_type, self.id, property_name)

    def set_data(self, property_name, value):
        if self._owner is None:
            self._temp_storage[property_name] = value
        else:
            self.owner.storage.update_data(self.obj_type, self.id, property_name, value)

    # def __eq__(self, other):
    #     if type(self) != type(other): return False
    #     # do not compare 'utterances' and 'conversations' in Speaker.__dict__. recursion loop will occur.
    #     self_keys = set(self.__dict__).difference(['_owner', 'meta', 'utterances', 'conversations'])
    #     other_keys = set(other.__dict__).difference(['_owner', 'meta', 'utterances', 'conversations'])
    #     return self_keys == other_keys and all([self.__dict__[k] == other.__dict__[k] for k in self_keys])

    def retrieve_meta(self, key: str):
        """
        Retrieves a value stored under the key of the metadata of corpus object
        :param key: name of metadata attribute
        :return: value
        """
        return self.meta.get(key, None)

    def add_meta(self, key: str, value) -> None:
        """
        Adds a key-value pair to the metadata of the corpus object
        :param key: name of metadata attribute
        :param value: value of metadata attribute
        :return: None
        """
        self.meta[key] = value

    def get_vector(
        self, vector_name: str, as_dataframe: bool = False, columns: Optional[List[str]] = None
    ):
        """
        Get the vector stored as `vector_name` for this object.
        :param vector_name: name of vector
        :param as_dataframe: whether to return the vector as a dataframe (True) or in its raw array form (False). False
            by default.
        :param columns: optional list of named columns of the vector to include. All columns returned otherwise. This
            parameter is only used if as_dataframe is set to True
        :return: a numpy / scipy array
        """
        if vector_name not in self.vectors:
            raise ValueError(
                "This {} has no vector stored as '{}'.".format(self.obj_type, vector_name)
            )

        return self.owner.get_vector_matrix(vector_name).get_vectors(
            ids=[self.id], as_dataframe=as_dataframe, columns=columns
        )

    def add_vector(self, vector_name: str):
        """
        Logs in the Corpus component object's internal vectors list that the component object has a vector row
        associated with it in the vector matrix named `vector_name`.
        Transformers that add vectors to the Corpus should use this to update the relevant component objects during
        the transform() step.
        :param vector_name: name of vector matrix
        :return: None
        """
        if vector_name not in self.vectors:
            self.vectors.append(vector_name)

    def has_vector(self, vector_name: str):
        return vector_name in self.vectors

    def delete_vector(self, vector_name: str):
        """
        Delete a vector associated with this Corpus component object.
        :param vector_name:
        :return: None
        """
        self.vectors.remove(vector_name)

    def to_dict(self):
        return {
            "id": self.id,
            "vectors": self.vectors,
            "meta": self.meta if type(self.meta) == dict else self.meta.to_dict(),
        }

    def __str__(self):
        return "{}(id: {}, vectors: {}, meta: {})".format(
            self.obj_type.capitalize(), self.id, self.vectors, self.meta
        )

    def __hash__(self):
        return hash(self.obj_type + str(self.id))

    def __repr__(self):
        copy = self.__dict__.copy()
        deleted_keys = [
            "utterances",
            "conversations",
            "user",
            "_root",
            "_utterance_ids",
            "_speaker_ids",
        ]
        for k in deleted_keys:
            if k in copy:
                del copy[k]

        to_delete = [k for k in copy if k.startswith("_")]
        to_add = {k[1:]: copy[k] for k in copy if k.startswith("_")}

        for k in to_delete:
            del copy[k]

        copy.update(to_add)

        try:
            return self.obj_type.capitalize() + "(" + str(copy) + ")"
        except (
            AttributeError
        ):  # for backwards compatibility when corpus objects are saved as binary data, e.g. wikiconv
            return "(" + str(copy) + ")"


class StorageManager(metaclass=ABCMeta):
    """
    Abstraction layer for the concrete representation of data and metadata
    within corpus components (e.g., Utterance text and timestamps). All requests
    to access or modify corpusComponent fields (with the exception of ID) are
    actually routed through one of StorageManager's concrete subclasses. Each
    subclass implements a storage backend that contains the actual data.
    """

    def __init__(self):
        # concrete data storage (i.e., collections) for each component type
        # this will be assigned in subclasses
        self.data = {"utterance": None, "conversation": None, "speaker": None, "meta": None}

    @abstractmethod
    def get_collection_ids(self, component_type: str):
        """
        Returns a list of all object IDs within the component_type collection
        """
        return NotImplemented

    @abstractmethod
    def has_data_for_component(self, component_type: str, component_id: str) -> bool:
        """
        Check if there is an existing entry for the component of type component_type
        with id component_id
        """
        return NotImplemented

    @abstractmethod
    def initialize_data_for_component(
        self, component_type: str, component_id: str, overwrite: bool = False, initial_value=None
    ):
        """
        Create a blank entry for a component of type component_type with id
        component_id. Will avoid overwriting any existing data unless the
        overwrite parameter is set to True.
        """
        return NotImplemented

    @abstractmethod
    def get_data(
        self,
        component_type: str,
        component_id: str,
        property_name: Optional[str] = None,
        index=None,
    ):
        """
        Retrieve the property data for the component of type component_type with
        id component_id. If property_name is specified return only the data for
        that property, otherwise return the dict containing all properties.
        Additionally, the expected type of the property to be fetched may be specified
        as a string; this is meant to be used for metadata in conjunction with the index.
        """
        return NotImplemented

    @abstractmethod
    def update_data(
        self,
        component_type: str,
        component_id: str,
        property_name: str,
        new_value,
        index=None,
    ):
        """
        Set or update the property data for the component of type component_type
        with id component_id. For metadata, the Python object type may also be
        specified, to be used in conjunction with the index.
        """
        return NotImplemented

    @abstractmethod
    def delete_data(
        self, component_type: str, component_id: str, property_name: Optional[str] = None
    ):
        """
        Delete a data entry from this StorageManager for the component of type
        component_type with id component_id. If property_name is specified
        delete only that property, otherwise delete the entire entry.
        """
        return NotImplemented

    @abstractmethod
    def clear_all_data(self):
        """
        Erase all data from this StorageManager (i.e., reset self.data to its
        initial empty state; Python will garbage-collect the now-unreferenced
        old data entries). This is used for cleanup after destructive Corpus
        operations.
        """
        return NotImplemented

    @abstractmethod
    def count_entries(self, component_type: str):
        """
        Count the number of entries held for the specified component type by
        this StorageManager instance
        """
        return NotImplemented

    def get_collection(self, component_type: str):
        if component_type not in self.data:
            raise ValueError(
                'component_type must be one of "utterance", "conversation", "speaker", or "meta".'
            )
        return self.data[component_type]

    def purge_obsolete_entries(self, utterance_ids, conversation_ids, speaker_ids, meta_ids):
        """
        Compare the entries in this StorageManager to the existing component ids
        provided as parameters, and delete any entries that are not found in the
        parameter ids.
        """
        ref_ids = {
            "utterance": set(utterance_ids),
            "conversation": set(conversation_ids),
            "speaker": set(speaker_ids),
            "meta": set(meta_ids),
        }
        for obj_type in self.data:
            for obj_id in self.get_collection_ids(obj_type):
                if obj_id not in ref_ids[obj_type]:
                    self.delete_data(obj_type, obj_id)

class Speaker(CorpusComponent):
    """
    Represents a single speaker in a dataset.
    :param id: id of the speaker.
    :type id: str
    :param utts: dictionary of utterances by the speaker, where key is utterance id
    :param convos: dictionary of conversations started by the speaker, where key is conversation id
    :param meta: arbitrary dictionary of attributes associated
        with the speaker.
    :type meta: dict
    :ivar id: id of the speaker.
    :ivar meta: A dictionary-like view object providing read-write access to
        speaker-level metadata.
    """

    def __init__(
        self,
        owner=None,
        id: str = None,
        utts=None,
        convos=None,
        meta: Optional[Dict] = None,
    ):
        super().__init__(obj_type="speaker", owner=owner, id=id, meta=meta)
        self.utterances = utts if utts is not None else dict()
        self.conversations = convos if convos is not None else dict()
        # self._split_attribs = set()
        # self._update_uid()

    # def identify_by_attribs(self, attribs: Collection) -> None:
    #     """Identify a speaker by a list of attributes. Sets which speaker info
    #     attributes should distinguish speakers of the same name in equality tests.
    #     For example, in the Supreme Court dataset, speakers are labeled with the
    #     current case id. Call this method with attribs = ["case"] to count
    #     the same person across different cases as different speakers.
    #
    #     By default, if this function is not called, speakers are identified by name only.
    #
    #     :param attribs: Collection of attribute names.
    #     :type attribs: Collection
    #     """
    #
    #     self._split_attribs = set(attribs)
    #     # self._update_uid()

    def _add_utterance(self, utt):
        self.utterances[utt.id] = utt

    def _add_conversation(self, convo):
        self.conversations[convo.id] = convo

    def get_utterance(self, ut_id: str):  # -> Utterance:
        """
        Get the Utterance with the specified Utterance id
        :param ut_id: The id of the Utterance
        :return: An Utterance object
        """
        return self.utterances[ut_id]

    def iter_utterances(self, selector=lambda utt: True):  # -> Generator[Utterance, None, None]:
        """
        Get utterances made by the Speaker, with an optional selector that selects for Utterances that
        should be included.
                :param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude).
                        By default, the selector includes all Utterances in the Corpus.
        :return: An iterator of the Utterances made by the speaker
        """
        for v in self.utterances.values():
            if selector(v):
                yield v

    def get_utterances_dataframe(self, selector=lambda utt: True, exclude_meta: bool = False):
        """
        Get a DataFrame of the Utterances made by the Speaker with fields and metadata attributes.
        Set an optional selector that filters for Utterances that should be included.
        Edits to the DataFrame do not change the corpus in any way.
        :param exclude_meta: whether to exclude metadata
        :param selector: a (lambda) function that takes a Utterance and returns True or False (i.e. include / exclude).
                By default, the selector includes all Utterances in the Corpus.
        :return: a pandas DataFrame
        """
        return get_utterances_dataframe(self, selector, exclude_meta)

    def get_utterance_ids(self, selector=lambda utt: True) -> List[str]:
        """
        :return: a List of the ids of Utterances made by the speaker
        """
        return list([utt.id for utt in self.iter_utterances(selector)])

    def get_conversation(self, cid: str):  # -> Conversation:
        """
        Get the Conversation with the specified Conversation id
        :param cid: The id of the Conversation
        :return: A Conversation object
        """
        return self.conversations[cid]

    def iter_conversations(
        self, selector=lambda convo: True
    ):  # -> Generator[Conversation, None, None]:
        """
        :return: An iterator of the Conversations that the speaker has participated in
        """
        for v in self.conversations.values():
            if selector(v):
                yield v

    def get_conversations_dataframe(self, selector=lambda convo: True, exclude_meta: bool = False):
        """
        Get a DataFrame of the Conversations the Speaker has participated in, with fields and metadata attributes.
        Set an optional selector that filters for Conversations that should be included. Edits to the DataFrame do not
        change the corpus in any way.
        :param exclude_meta: whether to exclude metadata
        :param selector: a (lambda) function that takes a Conversation and returns True or False (i.e. include / exclude).
            By default, the selector includes all Conversations in the Corpus.
        :return: a pandas DataFrame
        """
        return get_conversations_dataframe(self, selector, exclude_meta)

    def get_conversation_ids(self, selector=lambda convo: True) -> List[str]:
        """
        :return: a List of the ids of Conversations started by the speaker
        """
        return [convo.id for convo in self.iter_conversations(selector)]

    def print_speaker_stats(self):
        """
        Helper function for printing the number of Utterances made and Conversations participated in by the Speaker.
        :return: None (prints output)
        """
        print("Number of Utterances: {}".format(len(list(self.iter_utterances()))))
        print("Number of Conversations: {}".format(len(list(self.iter_conversations()))))

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Speaker):
            return False
        try:
            return self.id == other.id
        except AttributeError:
            return self.__dict__["_name"] == other.__dict__["_name"]

    def __str__(self):
        return "Speaker(id: {}, vectors: {}, meta: {})".format(
            repr(self.id), self.vectors, self.meta
        )


class Utterance(CorpusComponent):
    """Represents a single utterance in the dataset.
    :param id: the unique id of the utterance.
    :param speaker: the speaker giving the utterance.
    :param conversation_id: the id of the root utterance of the conversation.
    :param reply_to: id of the utterance this was a reply to.
    :param timestamp: timestamp of the utterance. Can be any
        comparable type.
    :param text: text of the utterance.
    :ivar id: the unique id of the utterance.
    :ivar speaker: the speaker giving the utterance.
    :ivar conversation_id: the id of the root utterance of the conversation.
    :ivar reply_to: id of the utterance this was a reply to.
    :ivar timestamp: timestamp of the utterance.
    :ivar text: text of the utterance.
    :ivar meta: A dictionary-like view object providing read-write access to
        utterance-level metadata.
    """

    def __init__(
        self,
        owner=None,
        id: Optional[str] = None,
        speaker: Optional[Speaker] = None,
        conversation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        timestamp: Optional[int] = None,
        text: str = "",
        meta: Optional[Dict] = None,
    ):
        # check arguments that have alternate naming due to backwards compatibility
        if speaker is None:
            raise ValueError("No Speaker found: Utterance must be initialized with a Speaker.")

        if conversation_id is not None and not isinstance(conversation_id, str):
            warn(
                "Utterance conversation_id must be a string: conversation_id of utterance with ID: {} "
                "has been casted to a string.".format(id)
            )
            conversation_id = str(conversation_id)
        if not isinstance(text, str):
            warn(
                "Utterance text must be a string: text of utterance with ID: {} "
                "has been casted to a string.".format(id)
            )
            text = "" if text is None else str(text)

        props = {
            "speaker_id": speaker.id,
            "conversation_id": conversation_id,
            "reply_to": reply_to,
            "timestamp": timestamp,
            "text": text,
        }
        super().__init__(obj_type="utterance", owner=owner, id=id, initial_data=props, meta=meta)
        self.speaker_ = speaker

    ############################################################################
    ## directly-accessible class properties (roughly equivalent to keys in the
    ## JSON, plus aliases for compatibility)
    ############################################################################

    def _get_speaker(self):
        return self.speaker_

    def _set_speaker(self, val):
        self.speaker_ = val
        self.set_data("speaker_id", self.speaker.id)

    speaker = property(_get_speaker, _set_speaker)

    def _get_conversation_id(self):
        return self.get_data("conversation_id")

    def _set_conversation_id(self, val):
        self.set_data("conversation_id", val)

    conversation_id = property(_get_conversation_id, _set_conversation_id)

    def _get_reply_to(self):
        return self.get_data("reply_to")

    def _set_reply_to(self, val):
        self.set_data("reply_to", val)

    reply_to = property(_get_reply_to, _set_reply_to)

    def _get_timestamp(self):
        return self.get_data("timestamp")

    def _set_timestamp(self, val):
        self.set_data("timestamp", val)

    timestamp = property(_get_timestamp, _set_timestamp)

    def _get_text(self):
        return self.get_data("text")

    def _set_text(self, val):
        self.set_data("text", val)

    text = property(_get_text, _set_text)

    ############################################################################
    ## end properties
    ############################################################################

    def get_conversation(self):
        """
        Get the Conversation (identified by Utterance.conversation_id) this Utterance belongs to
        :return: a Conversation object
        """
        return self.owner.get_conversation(self.conversation_id)

    def get_speaker(self):
        """
        Get the Speaker that made this Utterance.
        :return: a Speaker object
        """

        return self.speaker

    def to_dict(self):
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "reply_to": self.reply_to,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "text": self.text,
            "vectors": self.vectors,
            "meta": self.meta if type(self.meta) == dict else self.meta.to_dict(),
        }

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Utterance):
            return False
        try:
            return (
                self.id == other.id
                and (
                    self.conversation_id is None
                    or other.conversation_id is None
                    or self.conversation_id == other.conversation_id
                )
                and self.reply_to == other.reply_to
                and self.speaker == other.speaker
                and self.timestamp == other.timestamp
                and self.text == other.text
            )
        except AttributeError:  # for backwards compatibility with wikiconv
            return self.__dict__ == other.__dict__

    def __str__(self):
        return (
            "Utterance(id: {}, conversation_id: {}, reply-to: {}, "
            "speaker: {}, timestamp: {}, text: {}, vectors: {}, meta: {})".format(
                repr(self.id),
                self.conversation_id,
                self.reply_to,
                self.speaker,
                self.timestamp,
                repr(self.text),
                self.vectors,
                self.meta,
            )
        )


class ConvoKitMeta(MutableMapping, dict):
    """
    ConvoKitMeta is a dictlike object that stores the metadata attributes of a corpus component
    """

    def __init__(self, owner, convokit_index, obj_type, overwrite=False):
        self.owner = owner  # Corpus or CorpusComponent
        self.index: ConvoKitIndex = convokit_index
        self.obj_type = obj_type

        self._get_storage().initialize_data_for_component(
            "meta", self.storage_key, overwrite=overwrite
        )

    @property
    def storage_key(self) -> str:
        return f"{self.obj_type}_{self.owner.id}"

    def __getitem__(self, item):
        return self._get_storage().get_data(
            "meta", self.storage_key, item, self.index.get_index(self.obj_type)
        )

    def _get_storage(self):
        # special case for Corpus meta since that's the only time owner is not a CorpusComponent
        # since cannot directly import Corpus to check the type (circular import), as a proxy we
        # check for the obj_type attribute which is common to all CorpusComponent but not
        # present in Corpus
        if not hasattr(self.owner, "obj_type"):
            return self.owner.storage
        # self.owner -> CorpusComponent
        # self.owner.owner -> Corpus that owns the CorpusComponent (only Corpus has direct pointer to storage)
        return self.owner.owner.storage

    @staticmethod
    def _check_type_and_update_index(index, obj_type, key, value):
        if key not in index.indices[obj_type]:
            if isinstance(value, type(None)):  # new entry with None type means can't infer type yet
                index.create_new_index(obj_type, key=key)
            else:
                type_ = _optimized_type_check(value)
                index.update_index(obj_type, key=key, class_type=type_)
        else:
            # entry exists
            if not isinstance(value, type(None)):  # do not update index if value is None
                if index.get_index(obj_type)[key] != ["bin"]:  # if "bin" do no further checks
                    if str(type(value)) not in index.get_index(obj_type)[key]:
                        new_type = _optimized_type_check(value)

                        if new_type == "bin":
                            index.set_index(obj_type, key, "bin")
                        else:
                            index.update_index(obj_type, key, new_type)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            warn("Metadata attribute keys must be strings. Input key has been casted to a string.")
            key = str(key)

        if self.index.type_check:
            ConvoKitMeta._check_type_and_update_index(self.index, self.obj_type, key, value)
        self._get_storage().update_data(
            "meta", self.storage_key, key, value, self.index.get_index(self.obj_type)
        )

    def __delitem__(self, key):
        if self.obj_type == "corpus":
            self.index.del_from_index(self.obj_type, key)
            self._get_storage().delete_data("meta", self.storage_key, key)
        else:
            if self.index.lock_metadata_deletion[self.obj_type]:
                warn(
                    "For consistency in metadata attributes in Corpus component objects, deleting metadata attributes "
                    "from component objects individually is not allowed. "
                    "To delete this metadata attribute from all Corpus components of this type, "
                    "use corpus.delete_metadata(obj_type='{}', attribute='{}') instead.".format(
                        self.obj_type, key
                    )
                )
            else:
                self._get_storage().delete_data("meta", self.storage_key, key)

    def __iter__(self):
        return (
            self._get_storage()
            .get_data("meta", self.storage_key, index=self.index.get_index(self.obj_type))
            .__iter__()
        )

    def __len__(self):
        return (
            self._get_storage()
            .get_data("meta", self.storage_key, index=self.index.get_index(self.obj_type))
            .__len__()
        )

    def __contains__(self, x):
        return (
            self._get_storage()
            .get_data("meta", self.storage_key, index=self.index.get_index(self.obj_type))
            .__contains__(x)
        )

    def __repr__(self) -> str:
        return "ConvoKitMeta(" + self.to_dict().__repr__() + ")"

    def to_dict(self):
        return dict(
            self._get_storage().get_data(
                "meta", self.storage_key, index=self.index.get_index(self.obj_type)
            )
        )

    def reinitialize_from(self, other: Union["ConvoKitMeta", dict]):
        """
        Reinitialize this ConvoKitMeta instance with the data from other
        """
        if isinstance(other, ConvoKitMeta):
            other = {k: v for k, v in other.to_dict().items()}
        elif not isinstance(other, dict):
            raise TypeError(
                "ConvoKitMeta can only be reinitialized from a dict instance or another ConvoKitMeta"
            )
        self._get_storage().initialize_data_for_component(
            "meta", self.storage_key, overwrite=True, initial_value=other
        )


# _basic_types = {type(0), type(1.0), type("str"), type(True)}  # cannot include lists or dicts


# def _optimized_type_check(val):
#     # if type(obj)
#     if type(val) in _basic_types:
#         return str(type(val))
#     else:
#         try:
#             json.dumps(val)
#             return str(type(val))
#         except (TypeError, OverflowError):
#             return "bin"

DEFAULT_CONFIG_CONTENTS = (
    "# Default Storage Parameters\n"
    "db_host: localhost:27017\n"
    "data_directory: ~/.convokit/saved-corpora\n"
    "default_storage_mode: mem"
)

ENV_VARS = {"db_host": "CONVOKIT_DB_HOST", "default_storage_mode": "CONVOKIT_STORAGE_MODE"}


class ConvoKitConfig:
    """
    Utility class providing read-only access to the ConvoKit config file
    """

    def __init__(self, filename: Optional[str] = None):
        if filename is None:
            filename = os.path.expanduser("~/.convokit/config.yml")

        if not os.path.isfile(filename):
            convo_dir = os.path.dirname(filename)
            if not os.path.isdir(convo_dir):
                os.makedirs(convo_dir)
            with open(filename, "w") as f:
                print(
                    f"No configuration file found at {filename}; writing with contents: \n{DEFAULT_CONFIG_CONTENTS}"
                )
                f.write(DEFAULT_CONFIG_CONTENTS)
                self.config_contents = load(DEFAULT_CONFIG_CONTENTS, Loader=Loader)
        else:
            with open(filename, "r") as f:
                self.config_contents = load(f.read(), Loader=Loader)

    def _get_config_from_env_or_file(self, config_key: str, default_val):
        env_val = os.environ.get(ENV_VARS[config_key], None)
        if env_val is not None:
            # environment variable setting takes priority
            return env_val
        return self.config_contents.get(config_key, default_val)

    @property
    def db_host(self):
        return self._get_config_from_env_or_file("db_host", "localhost:27017")

    @property
    def data_directory(self):
        return self.config_contents.get("data_directory", "~/.convokit/saved-corpora")

    @property
    def default_storage_mode(self):
        return self._get_config_from_env_or_file("default_storage_mode", "mem")

class Conversation(CorpusComponent):
    """
    Represents a discrete subset of utterances in the dataset, connected by a reply-to chain.
    :param owner: The Corpus that this Conversation belongs to
    :param id: The unique ID of this Conversation
    :param utterances: A list of the IDs of the Utterances in this Conversation
    :param meta: Table of initial values for conversation-level metadata
    :ivar id: the ID of the Conversation
    :ivar meta: A dictionary-like view object providing read-write access to
        conversation-level metadata.
    """

    def __init__(
        self,
        owner,
        id: Optional[str] = None,
        utterances: Optional[List[str]] = None,
        meta: Optional[Dict] = None,
    ):
        super().__init__(obj_type="conversation", owner=owner, id=id, meta=meta)
        self._owner = owner
        self._utterance_ids: List[str] = utterances
        self._speaker_ids = None
        self.tree: Optional[UtteranceNode] = None

    def _add_utterance(self, utt: Utterance):
        self._utterance_ids.append(utt.id)
        self._speaker_ids = None
        self.tree = None

    def get_utterance_ids(self) -> List[str]:
        """Produces a list of the unique IDs of all utterances in the
        Conversation, which can be used in calls to get_utterance() to retrieve
        specific utterances. Provides no ordering guarantees for the list.
        :return: a list of IDs of Utterances in the Conversation
        """
        # pass a copy of the list
        return self._utterance_ids[:]

    def get_utterance(self, ut_id: str) -> Utterance:
        """Looks up the Utterance associated with the given ID. Raises a
        KeyError if no utterance by that ID exists.
        :return: the Utterance with the given ID
        """
        # delegate to the owner Corpus since Conversation does not itself own
        # any Utterances
        return self._owner.get_utterance(ut_id)

    def iter_utterances(
        self, selector: Callable[[Utterance], bool] = lambda utt: True
    ) -> Generator[Utterance, None, None]:
        """
        Get utterances in the Corpus, with an optional selector that filters for Utterances that should be included.
        :param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude).
                        By default, the selector includes all Utterances in the Conversation.
                :return: a generator of Utterances
        """
        for ut_id in self._utterance_ids:
            utt = self._owner.get_utterance(ut_id)
            if selector(utt):
                yield utt

    def get_utterances_dataframe(
        self, selector: Callable[[Utterance], bool] = lambda utt: True, exclude_meta: bool = False
    ):
        """
        Get a DataFrame of the Utterances in the Conversation with fields and metadata attributes.
                Set an optional selector that filters for Utterances that should be included.
                Edits to the DataFrame do not change the corpus in any way.
        :param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude).
                        By default, the selector includes all Utterances in the Conversation.
        :param exclude_meta: whether to exclude metadata
        :return: a pandas DataFrame
        """
        return get_utterances_dataframe(self, selector, exclude_meta)

    def get_speaker_ids(self) -> List[str]:
        """
        Produces a list of ids of all speakers in the Conversation, which can be used in calls to get_speaker()
        to retrieve specific speakers. Provides no ordering guarantees for the list.
        :return: a list of speaker ids
        """
        if self._speaker_ids is None:
            # first call to get_speaker_ids or iter_speakers; precompute cached list of speaker ids
            self._speaker_ids = set()
            for ut_id in self._utterance_ids:
                ut = self._owner.get_utterance(ut_id)
                self._speaker_ids.add(ut.speaker.id)
        return list(self._speaker_ids)

    def get_speaker(self, speaker_id: str) -> Speaker:
        """
        Looks up the Speaker with the given name. Raises a KeyError if no speaker
        with that name exists.
        :return: the Speaker with the given speaker_id
        """
        # delegate to the owner Corpus since Conversation does not itself own
        # any Utterances
        return self._owner.get_speaker(speaker_id)

    def iter_speakers(
        self, selector: Callable[[Speaker], bool] = lambda speaker: True
    ) -> Generator[Speaker, None, None]:
        """
        Get Speakers that have participated in the Conversation, with an optional selector that filters for Speakers
        that should be included.
                :param selector: a (lambda) function that takes a Speaker and returns True or False (i.e. include / exclude).
                        By default, the selector includes all Speakers in the Conversation.
                :return: a generator of Speakers
        """
        if self._speaker_ids is None:
            # first call to get_ids or iter_speakers; precompute cached list of speaker ids
            self._speaker_ids = set()
            for ut_id in self._utterance_ids:
                ut = self._owner.get_utterance(ut_id)
                self._speaker_ids.add(ut.speaker.id)
        for speaker_id in self._speaker_ids:
            speaker = self._owner.get_speaker(speaker_id)
            if selector(speaker):
                yield speaker

    def get_speakers_dataframe(
        self,
        selector: Optional[Callable[[Speaker], bool]] = lambda utt: True,
        exclude_meta: bool = False,
    ):
        """
        Get a DataFrame of the Speakers that have participated in the Conversation with fields and metadata attributes,
        with an optional selector that filters Speakers that should be included.
        Edits to the DataFrame do not change the corpus in any way.
                :param exclude_meta: whether to exclude metadata
                :param selector: selector: a (lambda) function that takes a Speaker and returns True or False
                        (i.e. include / exclude). By default, the selector includes all Speakers in the Conversation.
                :return: a pandas DataFrame
        """
        return get_speakers_dataframe(self, selector, exclude_meta)

    def print_conversation_stats(self):
        """
        Helper function for printing the number of Utterances and Spekaers in the Conversation.
        :return: None (prints output)
        """
        print("Number of Utterances: {}".format(len(list(self.iter_utterances()))))
        print("Number of Speakers: {}".format(len(list(self.iter_speakers()))))

    def get_chronological_speaker_list(
        self, selector: Callable[[Speaker], bool] = lambda speaker: True
    ):
        """
        Get the speakers in the conversation sorted in chronological order (speakers may appear more than once)
        :param selector: (lambda) function for which speakers should be included; all speakers are included by default
        :return: list of speakers for each chronological utterance
        """
        try:
            chrono_utts = sorted(list(self.iter_utterances()), key=lambda utt: utt.timestamp)
            return [utt.speaker for utt in chrono_utts if selector(utt.speaker)]
        except TypeError as e:
            raise ValueError(str(e) + "\nUtterance timestamps may not have been set correctly.")

    def check_integrity(self, verbose: bool = True) -> bool:
        """
        Check the integrity of this Conversation; i.e. do the constituent utterances form a complete reply-to chain?
        :param verbose: whether to print errors indicating the problems with the Conversation
        :return: True if the conversation structure is complete else False
        """
        if verbose:
            print("Checking reply-to chain of Conversation", self.id)
        utt_reply_tos = {utt.id: utt.reply_to for utt in self.iter_utterances()}
        target_utt_ids = set(list(utt_reply_tos.values()))
        speaker_utt_ids = set(list(utt_reply_tos.keys()))
        root_utt_id = target_utt_ids - speaker_utt_ids  # There should only be 1 root_utt_id: None

        if len(root_utt_id) != 1:
            if verbose:
                for utt_id in root_utt_id:
                    if utt_id is not None:
                        warn("ERROR: Missing utterance {}".format(utt_id))
            return False
        else:
            root_id = list(root_utt_id)[0]
            if root_id is not None:
                if verbose:
                    warn("ERROR: Missing utterance {}".format(root_id))
                return False

        # sanity check
        utts_replying_to_none = 0
        for utt in self.iter_utterances():
            if utt.reply_to is None:
                utts_replying_to_none += 1

        if utts_replying_to_none > 1:
            if verbose:
                warn("ERROR: Found more than one Utterance replying to None.")
            return False

        circular = [
            utt_id for utt_id, utt_reply_to in utt_reply_tos.items() if utt_id == utt_reply_to
        ]
        if len(circular) > 0:
            if verbose:
                warn(
                    "ERROR: Found utterances with .reply_to pointing to themselves: {}".format(
                        circular
                    )
                )
            return False

        if verbose:
            print("No issues found.\n")
        return True

    def initialize_tree_structure(self):
        if not self.check_integrity(verbose=False):
            raise ValueError(
                "Conversation {} reply-to chain does not form a valid tree.".format(self.id)
            )

        root_node_id = None
        # Find root node
        for utt in self.iter_utterances():
            if utt.reply_to is None:
                root_node_id = utt.id

        parent_to_children_ids = defaultdict(list)
        for utt in self.iter_utterances():
            parent_to_children_ids[utt.reply_to].append(utt.id)

        wrapped_utts = {utt.id: UtteranceNode(utt) for utt in self.iter_utterances()}

        for parent_id, wrapped_utt in wrapped_utts.items():
            wrapped_utt.set_children(
                [wrapped_utts[child_id] for child_id in parent_to_children_ids[parent_id]]
            )

        self.tree = wrapped_utts[root_node_id]

    def traverse(self, traversal_type: str, as_utterance: bool = True):
        """
        Traverse through the Conversation tree structure in a breadth-first search ('bfs'), depth-first search (dfs),
        pre-order ('preorder'), or post-order ('postorder') way.
        :param traversal_type: dfs, bfs, preorder, or postorder
        :param as_utterance: whether the iterator should yield the utterance (True) or the utterance node (False)
        :return: an iterator of the utterances or utterance nodes
        """
        if self.tree is None:
            self.initialize_tree_structure()
            if self.tree is None:
                raise ValueError(
                    "Failed to traverse because Conversation reply-to chain does not form a valid tree."
                )

        traversals = {
            "bfs": self.tree.bfs_traversal,
            "dfs": self.tree.dfs_traversal,
            "preorder": self.tree.pre_order,
            "postorder": self.tree.post_order,
        }

        for utt_node in traversals[traversal_type]():
            yield utt_node.utt if as_utterance else utt_node

    def get_subtree(self, root_utt_id):
        """
        Get the utterance node of the specified input id
        :param root_utt_id: id of the root node that the subtree starts from
        :return: UtteranceNode object
        """
        if self.tree is None:
            self.initialize_tree_structure()
            if self.tree is None:
                raise ValueError(
                    "Failed to traverse because Conversation reply-to chain does not form a valid tree."
                )

        for utt_node in self.tree.bfs_traversal():
            if utt_node.utt.id == root_utt_id:
                return utt_node

    def get_longest_paths(self) -> List[List[Utterance]]:
        """
        Finds the Utterances form the longest path (i.e. root to leaf) in the Conversation tree.
        If there are multiple paths with tied lengths, returns all of them as a list of lists. If only one such path
        exists, a list containing a single list of Utterances is returned.
        :return: a list of lists of Utterances
        """
        if self.tree is None:
            self.initialize_tree_structure()
            if self.tree is None:
                raise ValueError(
                    "Failed to traverse because Conversation reply-to chain does not form a valid tree."
                )

        paths = self.get_root_to_leaf_paths()
        max_len = max(len(path) for path in paths)

        return [p for p in paths if len(p) == max_len]

    def _print_convo_helper(
        self,
        root: str,
        indent: int,
        reply_to_dict: Dict[str, str],
        utt_info_func: Callable[[Utterance], str],
        limit=None,
    ) -> None:
        """
        Helper function for print_conversation_structure()
        """
        if limit is not None:
            if self.get_utterance(root).meta["order"] > limit:
                return
        print(" " * indent + utt_info_func(self.get_utterance(root)))
        children_utt_ids = [k for k, v in reply_to_dict.items() if v == root]
        for child_utt_id in children_utt_ids:
            self._print_convo_helper(
                root=child_utt_id,
                indent=indent + 4,
                reply_to_dict=reply_to_dict,
                utt_info_func=utt_info_func,
                limit=limit,
            )

    def print_conversation_structure(
        self,
        utt_info_func: Callable[[Utterance], str] = lambda utt: utt.speaker.id,
        limit: int = None,
    ) -> None:
        """
        Prints an indented representation of utterances in the Conversation with conversation reply-to structure
        determining the indented level. The details of each utterance to be printed can be configured.
        If limit is set to a value other than None, this will annotate utterances with an 'order' metadata indicating
        their temporal order in the conversation, where the first utterance in the conversation is annotated with 1.
        :param utt_info_func: callable function taking an utterance as input and returning a string of the desired
            utterance information. By default, this is a lambda function returning the utterance's speaker's id
        :param limit: maximum number of utterances to print out. if k, this includes the first k utterances.
        :return: None. Prints to stdout.
        """
        if not self.check_integrity(verbose=False):
            raise ValueError(
                "Could not print conversation structure: The utterance reply-to chain is broken. "
                "Try check_integrity() to diagnose the problem."
            )

        if limit is not None:
            assert isinstance(limit, int)
            for idx, utt in enumerate(self.get_chronological_utterance_list()):
                utt.meta["order"] = idx + 1

        root_utt_id = [utt for utt in self.iter_utterances() if utt.reply_to is None][0].id
        reply_to_dict = {utt.id: utt.reply_to for utt in self.iter_utterances()}

        self._print_convo_helper(
            root=root_utt_id,
            indent=0,
            reply_to_dict=reply_to_dict,
            utt_info_func=utt_info_func,
            limit=limit,
        )

    def get_utterances_dataframe(self, selector=lambda utt: True, exclude_meta: bool = False):
        """
        Get a DataFrame of the Utterances in the COnversation with fields and metadata attributes.
        Set an optional selector that filters Utterances that should be included.
        Edits to the DataFrame do not change the corpus in any way.
        :param exclude_meta: whether to exclude metadata
        :param selector: a (lambda) function that takes a Utterance and returns True or False (i.e. include / exclude).
                By default, the selector includes all Utterances in the Conversation.
        :return: a pandas DataFrame
        """
        return get_utterances_dataframe(self, selector, exclude_meta)

    def get_chronological_utterance_list(
        self, selector: Callable[[Utterance], bool] = lambda utt: True
    ):
        """
        Get the utterances in the conversation sorted in increasing order of timestamp
        :param selector: function for which utterances should be included; all utterances are included by default
        :return: list of utterances, sorted by timestamp
        """
        try:
            return sorted(
                [utt for utt in self.iter_utterances(selector)], key=lambda utt: utt.timestamp
            )
        except TypeError as e:
            raise ValueError(str(e) + "\nUtterance timestamps may not have been set correctly.")

    def _get_path_from_leaf_to_root(
        self, leaf_utt: Utterance, root_utt: Utterance
    ) -> List[Utterance]:
        """
        Helper function for get_root_to_leaf_paths, which returns the path for a given leaf_utt and root_utt
        """
        if leaf_utt == root_utt:
            return [leaf_utt]
        path = [leaf_utt]
        root_id = root_utt.id
        while leaf_utt.reply_to != root_id:
            path.append(self.get_utterance(leaf_utt.reply_to))
            leaf_utt = path[-1]
        path.append(root_utt)
        return path[::-1]

    def get_root_to_leaf_paths(self) -> List[List[Utterance]]:
        """
        Get the paths (stored as a list of lists of utterances) from the root to each of the leaves
        in the conversational tree
        :return: List of lists of Utterances
        """
        if not self.check_integrity(verbose=False):
            raise ValueError(
                "Conversation failed integrity check. "
                "It is either missing an utterance in the reply-to chain and/or has multiple root nodes. "
                "Run check_integrity() to diagnose issues."
            )

        utt_reply_tos = {utt.id: utt.reply_to for utt in self.iter_utterances()}
        target_utt_ids = set(list(utt_reply_tos.values()))
        speaker_utt_ids = set(list(utt_reply_tos.keys()))
        root_utt_id = target_utt_ids - speaker_utt_ids  # There should only be 1 root_utt_id: None
        assert len(root_utt_id) == 1
        root_utt = [utt for utt in self.iter_utterances() if utt.reply_to is None][0]
        leaf_utt_ids = speaker_utt_ids - target_utt_ids

        paths = [
            self._get_path_from_leaf_to_root(self.get_utterance(leaf_utt_id), root_utt)
            for leaf_utt_id in leaf_utt_ids
        ]
        return paths

    @staticmethod
    def generate_default_conversation_id(utterance_id):
        return f"__default_conversation__{utterance_id}"

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Conversation):
            return False
        return self.id == other.id and set(self._utterance_ids) == set(other._utterance_ids)

    def __str__(self):
        return "Conversation('id': {}, 'utterances': {}, 'meta': {})".format(
            repr(self.id), self._utterance_ids, self.meta
        )

class Corpus:
    """
    Represents a dataset, which can be loaded from a folder or constructed from a list of utterances.
    :param filename: Path to a folder containing a Corpus or to an utterances.jsonl / utterances.json file to load
    :param utterances: list of utterances to initialize Corpus from
    :param preload_vectors: list of names of vectors to be preloaded from directory; by default,
        no vectors are loaded but can be loaded any time after corpus initialization (i.e. vectors are lazy-loaded).
    :param utterance_start_index: if loading from directory and the corpus folder contains utterances.jsonl, specify the
        line number (zero-indexed) to begin parsing utterances from
    :param utterance_end_index: if loading from directory and the corpus folder contains utterances.jsonl, specify the
        line number (zero-indexed) of the last utterance to be parsed.
    :param merge_lines: whether to merge adjacent lines from same speaker if multiple consecutive utterances belong to
        the same conversation.
    :param exclude_utterance_meta: utterance metadata to be ignored
    :param exclude_conversation_meta: conversation metadata to be ignored
    :param exclude_speaker_meta: speaker metadata to be ignored
    :param exclude_overall_meta: overall metadata to be ignored
    :param disable_type_check: whether to do type checking when loading the Corpus from a directory.
        Type-checking ensures that the ConvoKitIndex is initialized correctly. However, it may be unnecessary if the
        index.json is already accurate and disabling it will allow for a faster corpus load. This parameter is set to
        True by default, i.e. type-checking is not carried out.
    :ivar meta_index: index of Corpus metadata
    :ivar vectors: the vectors stored in the Corpus
    :ivar corpus_dirpath: path to the directory the corpus was loaded from
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        utterances: Optional[List[Utterance]] = None,
        db_collection_prefix: Optional[str] = None,
        db_host: Optional[str] = None,
        preload_vectors: List[str] = None,
        utterance_start_index: int = None,
        utterance_end_index: int = None,
        merge_lines: bool = False,
        exclude_utterance_meta: Optional[List[str]] = None,
        exclude_conversation_meta: Optional[List[str]] = None,
        exclude_speaker_meta: Optional[List[str]] = None,
        exclude_overall_meta: Optional[List[str]] = None,
        disable_type_check=True,
        storage_type: Optional[str] = None,
        storage: Optional[StorageManager] = None,
    ):
        self.config = ConvoKitConfig()
        self.corpus_dirpath = get_corpus_dirpath(filename)

        # configure corpus ID (optional for mem mode, required for DB mode)
        if storage_type is None:
            storage_type = self.config.default_storage_mode
        if db_collection_prefix is None and filename is None and storage_type == "db":
            db_collection_prefix = create_safe_id()
            warn(
                "You are in DB mode, but no collection prefix was specified and no filename was given from which to infer one."
                "Will use a randomly generated unique prefix " + db_collection_prefix
            )
        self.id = get_corpus_id(db_collection_prefix, filename, storage_type)
        self.storage_type = storage_type
        self.storage = initialize_storage(self, storage, storage_type, db_host)

        self.meta_index = ConvoKitIndex(self)
        self.meta = ConvoKitMeta(self, self.meta_index, "corpus")

        # private storage
        self._vector_matrices = dict()

        convos_data = defaultdict(dict)
        if exclude_utterance_meta is None:
            exclude_utterance_meta = []
        if exclude_conversation_meta is None:
            exclude_conversation_meta = []
        if exclude_speaker_meta is None:
            exclude_speaker_meta = []
        if exclude_overall_meta is None:
            exclude_overall_meta = []

        if filename is not None and storage_type == "db":
            # JSON-to-DB construction mode uses a specialized code branch, which
            # optimizes for this use case by using direct batch insertions into the
            # DB rather than going through the StorageManager, hence improving
            # efficiency.

            with open(os.path.join(filename, "index.json"), "r") as f:
                idx_dict = json.load(f)
                self.meta_index.update_from_dict(idx_dict)

            # populate the DB with the contents of the source file
            ids_in_db = populate_db_from_file(
                filename,
                self.storage.db,
                self.id,
                self.meta_index,
                utterance_start_index,
                utterance_end_index,
                exclude_utterance_meta,
                exclude_conversation_meta,
                exclude_speaker_meta,
                exclude_overall_meta,
            )

            # with the StorageManager's DB now populated, initialize the corresponding
            # CorpusComponent instances.
            init_corpus_from_storage_manager(self, ids_in_db)

            self.meta_index.enable_type_check()
            # load preload_vectors
            if preload_vectors is not None:
                for vector_name in preload_vectors:
                    matrix = ConvoKitMatrix.from_dir(self.corpus_dirpath, vector_name)
                    if matrix is not None:
                        self._vector_matrices[vector_name] = matrix

            if merge_lines:
                self.utterances = merge_utterance_lines(self.utterances)
        else:
            # Construct corpus from file or directory
            if filename is not None:
                if disable_type_check:
                    self.meta_index.disable_type_check()
                if os.path.isdir(filename):
                    utterances = load_utterance_info_from_dir(
                        filename, utterance_start_index, utterance_end_index, exclude_utterance_meta
                    )

                    speakers_data = load_speakers_data_from_dir(filename, exclude_speaker_meta)
                    convos_data = load_convos_data_from_dir(filename, exclude_conversation_meta)
                    load_corpus_meta_from_dir(filename, self.meta, exclude_overall_meta)

                    with open(os.path.join(filename, "index.json"), "r") as f:
                        idx_dict = json.load(f)
                        self.meta_index.update_from_dict(idx_dict)

                    # unpack all binary data
                    unpack_all_binary_data(
                        filename=filename,
                        meta_index=self.meta_index,
                        meta=self.meta,
                        utterances=utterances,
                        speakers_data=speakers_data,
                        convos_data=convos_data,
                        exclude_utterance_meta=exclude_utterance_meta,
                        exclude_speaker_meta=exclude_speaker_meta,
                        exclude_conversation_meta=exclude_conversation_meta,
                        exclude_overall_meta=exclude_overall_meta,
                    )

                else:
                    speakers_data = defaultdict(dict)
                    convos_data = defaultdict(dict)
                    utterances = load_from_utterance_file(
                        filename, utterance_start_index, utterance_end_index
                    )

                self.utterances = dict()
                self.speakers = dict()

                initialize_speakers_and_utterances_objects(self, utterances, speakers_data)

                self.meta_index.enable_type_check()

                # load preload_vectors
                if preload_vectors is not None:
                    for vector_name in preload_vectors:
                        matrix = ConvoKitMatrix.from_dir(self.corpus_dirpath, vector_name)
                        if matrix is not None:
                            self._vector_matrices[vector_name] = matrix

            elif utterances is not None:  # Construct corpus from utterances list
                self.speakers = {utt.speaker.id: utt.speaker for utt in utterances}
                self.utterances = {utt.id: utt for utt in utterances}
                for speaker in self.speakers.values():
                    speaker.owner = self
                for utt in self.utterances.values():
                    utt.owner = self

            if merge_lines:
                self.utterances = merge_utterance_lines(self.utterances)

            if disable_type_check:
                self.meta_index.disable_type_check()
            # if corpus is nonempty (check for self.utterances), construct the conversation
            # data from the utterance list
            if hasattr(self, "utterances"):
                self.conversations = initialize_conversations(
                    self, convos_data, fill_missing_convo_ids=True
                )
                self.meta_index.enable_type_check()
                self.update_speakers_data()
    
    def update_speakers_data(self) -> None:
        """
        Updates the conversation and utterance lists of every Speaker in the Corpus
        :return: None
        """
        speakers_utts = defaultdict(list)
        speakers_convos = defaultdict(list)

        for utt in self.iter_utterances():
            speakers_utts[utt.speaker.id].append(utt)

        for convo in self.iter_conversations():
            for utt in convo.iter_utterances():
                speakers_convos[utt.speaker.id].append(convo)

        for speaker in self.iter_speakers():
            speaker.utterances = {utt.id: utt for utt in speakers_utts[speaker.id]}
            speaker.conversations = {convo.id: convo for convo in speakers_convos[speaker.id]}
    
    def iter_utterances(
        self, selector: Optional[Callable[[Utterance], bool]] = lambda utt: True
    ) -> Generator[Utterance, None, None]:
        """
        Get utterances in the Corpus, with an optional selector that filters for Utterances that should be included.
        :param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude).
            By default, the selector includes all Utterances in the Corpus.
        :return: a generator of Utterances
        """
        for v in self.utterances.values():
            if selector(v):
                yield v

    def iter_conversations(
        self, selector: Optional[Callable[[Conversation], bool]] = lambda convo: True
    ) -> Generator[Conversation, None, None]:
        """
        Get conversations in the Corpus, with an optional selector that filters for Conversations that should be included
        :param selector: a (lambda) function that takes a Conversation and returns True or False (i.e. include / exclude).
            By default, the selector includes all Conversations in the Corpus.
        :return: a generator of Conversations
        """
        for v in self.conversations.values():
            if selector(v):
                yield v

    def get_utterance(self, utt_id: str) -> Utterance:
        """
        Gets Utterance of the specified id from the corpus
        :param utt_id: id of Utterance
        :return: Utterance
        """
        return self.utterances[utt_id]

    def get_conversation(self, convo_id: str) -> Conversation:
        """
        Gets Conversation of the specified id from the corpus
        :param convo_id: id of Conversation
        :return: Conversation
        """
        return self.conversations[convo_id]

    def get_speaker(self, speaker_id: str) -> Speaker:
        """
        Gets Speaker of the specified id from the corpus
        :param speaker_id: id of Speaker
        :return: Speaker
        """
        return self.speakers[speaker_id]

    def get_object(self, obj_type: str, oid: str):
        """
        General Corpus object getter. Gets Speaker / Utterance / Conversation of specified id from the Corpus
        :param obj_type: "speaker", "utterance", or "conversation"
        :param oid: object id
        :return: Corpus object of specified object type with specified object id
        """
        assert obj_type in ["speaker", "utterance", "conversation"]
        if obj_type == "speaker":
            return self.get_speaker(oid)
        elif obj_type == "utterance":
            return self.get_utterance(oid)
        else:
            return self.get_conversation(oid)

    def iter_speakers(
        self, selector: Optional[Callable[[Speaker], bool]] = lambda speaker: True
    ) -> Generator[Speaker, None, None]:
        """
        Get Speakers in the Corpus, with an optional selector that filters for Speakers that should be included
        :param selector: a (lambda) function that takes a Speaker and returns True or False (i.e. include / exclude).
            By default, the selector includes all Speakers in the Corpus.
        :return: a generator of Speakers
        """

        for speaker in self.speakers.values():
            if selector(speaker):
                yield speaker

    @property
    def vectors(self):
        return self.meta_index.vectors

    @vectors.setter
    def vectors(self, new_vectors):
        if not isinstance(new_vectors, type(["stringlist"])):
            raise ValueError(
                "The preload_vectors being set should be a list of strings, "
                "where each string is the name of a vector matrix."
            )
        self.meta_index.vectors = new_vectors

    def dump(
        self,
        name: str,
        base_path: Optional[str] = None,
        exclude_vectors: List[str] = None,
        force_version: int = None,
        overwrite_existing_corpus: bool = False,
        fields_to_skip=None,
    ) -> None:
        """
        Dumps the corpus and its metadata to disk. Optionally, set `force_version` to a desired integer version number,
        otherwise the version number is automatically incremented.
        :param name: name of corpus
        :param base_path: base directory to save corpus in (None to save to a default directory)
        :param exclude_vectors: list of names of vector matrices to exclude from the dumping step. By default; all
            vector matrices that belong to the Corpus (whether loaded or not) are dumped.
        :param force_version: version number to set for the dumped corpus
        :param overwrite_existing_corpus: if True, save to the path you loaded the corpus from, overriding the original corpus.
        :param fields_to_skip: a dictionary of {object type: list of metadata attributes to omit when writing to disk}. object types can be one of "speaker", "utterance", "conversation", "corpus".
        """
        if fields_to_skip is None:
            fields_to_skip = dict()
        dir_name = name
        if base_path is not None and overwrite_existing_corpus:
            raise ValueError("Not allowed to specify both base_path and overwrite_existing_corpus!")
        if overwrite_existing_corpus and self.corpus_dirpath is None:
            raise ValueError(
                "Cannot use save to existing path on Corpus generated from utterance list!"
            )
        if not overwrite_existing_corpus:
            if base_path is None:
                base_path = os.path.expanduser("~/.convokit/")
                if not os.path.exists(base_path):
                    os.mkdir(base_path)
                base_path = os.path.join(base_path, "saved-corpora/")
                if not os.path.exists(base_path):
                    os.mkdir(base_path)
            dir_name = os.path.join(base_path, dir_name)
        else:
            dir_name = os.path.join(self.corpus_dirpath)

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        # dump speakers, conversations, utterances
        dump_corpus_component(
            self, dir_name, "speakers.json", "speaker", "speaker", exclude_vectors, fields_to_skip
        )
        dump_corpus_component(
            self,
            dir_name,
            "conversations.json",
            "conversation",
            "convo",
            exclude_vectors,
            fields_to_skip,
        )
        dump_utterances(self, dir_name, exclude_vectors, fields_to_skip)

        # dump corpus
        with open(os.path.join(dir_name, "corpus.json"), "w") as f:
            d_bin = defaultdict(list)
            meta_up = dump_helper_bin(self.meta, d_bin, fields_to_skip.get("corpus", None))

            json.dump(meta_up, f)
            for name, l_bin in d_bin.items():
                with open(os.path.join(dir_name, name + "-overall-bin.p"), "wb") as f_pk:
                    pickle.dump(l_bin, f_pk)

        # dump index
        with open(os.path.join(dir_name, "index.json"), "w") as f:
            json.dump(
                self.meta_index.to_dict(
                    exclude_vectors=exclude_vectors, force_version=force_version
                ),
                f,
            )

        # dump vectors
        if exclude_vectors is not None:
            vectors_to_dump = [v for v in self.vectors if v not in set(exclude_vectors)]
        else:
            vectors_to_dump = self.vectors
        for vector_name in vectors_to_dump:
            if vector_name in self._vector_matrices:
                self._vector_matrices[vector_name].dump(dir_name)
            else:
                src = os.path.join(self.corpus_dirpath, "vectors.{}.p".format(vector_name))
                dest = os.path.join(dir_name, "vectors.{}.p".format(vector_name))
                shutil.copy(src, dest)

    # with open(os.path.join(dir_name, "processed_text.index.json"), "w") as f:
    #     json.dump(list(self.processed_text.keys()), f)

    def get_object_ids(
        self,
        obj_type: str,
        selector: Callable[[Union[Speaker, Utterance, Conversation]], bool] = lambda obj: True,
    ):
        """
        Get a list of ids of Corpus objects of the specified type in the Corpus, with an optional selector that filters for objects that should be included
        :param obj_type: "speaker", "utterance", or "conversation"
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude).
            By default, the selector includes all objects of the specified type in the Corpus.
        :return: list of Corpus object ids
        """
        assert obj_type in ["speaker", "utterance", "conversation"]
        return [obj.id for obj in self.iter_objs(obj_type, selector)]

    def iter_objs(
        self,
        obj_type: str,
        selector: Callable[[Union[Speaker, Utterance, Conversation]], bool] = lambda obj: True,
    ):
        """
        Get Corpus objects of specified type from the Corpus, with an optional selector that filters for Corpus object that should be included
        :param obj_type: "speaker", "utterance", or "conversation"
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude).
            By default, the selector includes all objects of the specified type in the Corpus.
        :return: a generator of Speakers
        """

        assert obj_type in ["speaker", "utterance", "conversation"]
        obj_iters = {
            "conversation": self.iter_conversations,
            "speaker": self.iter_speakers,
            "utterance": self.iter_utterances,
        }

        return obj_iters[obj_type](selector)

    @staticmethod
    def from_pandas(
        utterances_df: DataFrame,
        speakers_df: Optional[DataFrame] = None,
        conversations_df: Optional[DataFrame] = None,
    ) -> "Corpus":
        """
        Generates a Corpus from utterances, speakers, and conversations dataframes.
        For each dataframe, if the 'id' column is absent, the dataframe index will be used as the id.
        Metadata should be denoted with a 'meta.<key>' column in the dataframe. For example, if an utterance is to have
        a metadata key 'score', then the 'meta.score' column must be present in dataframe.
        `speakers_df` and `conversations_df` are optional, as their IDs can be inferred from `utterances_df`, and so
        their main purpose is to hold speaker / conversation metadata. They should only be included if there exists
        metadata for the speakers / conversations respectively.
        Metadata values that are not basic Python data structures (i.e. lists, dicts, tuples) may be included in the
        dataframes but may lead to unexpected behavior, depending on how `pandas` serializes / deserializes those values.
        Note that as metadata can be added to the Corpus after it is constructed, there is no need to include all
        metadata keys in the dataframe if it would be inconvenient.
        :param utterances_df: utterances data in a pandas Dataframe, all primary data fields expected, with metadata optional
        :param speakers_df: (optional) speakers data in a pandas Dataframe
        :param conversations_df: (optional) conversations data in a pandas Dataframe
        :return: Corpus constructed from the dataframe(s)
        """
        columns = ["speaker", "id", "timestamp", "conversation_id", "reply_to", "text"]

        for df_type, df in [
            ("utterances", utterances_df),
            ("conversations", conversations_df),
            ("speakers", speakers_df),
        ]:
            if df is None:
                continue
            if "id" not in df.columns:
                print(
                    f"ID column is not present in {df_type} dataframe, generated ID column from dataframe index..."
                )
                df["id"] = df.index

        # checking if dataframes contain their respective required columns
        assert (
            pd.Series(columns).isin(utterances_df.columns).all()
        ), "Utterances dataframe must contain all primary data fields"

        utterance_meta_cols = extract_meta_from_df(utterances_df)

        utterance_list = []
        for index, row in tqdm(utterances_df.iterrows()):
            if utterance_meta_cols:
                metadata = {}
                for meta_col in utterance_meta_cols:
                    metadata[meta_col] = row["meta." + meta_col]
            else:
                metadata = None

            # adding utterance in utterance list
            reply_to = None if row["reply_to"] == "None" else row["reply_to"]
            utterance_list.append(
                Utterance(
                    id=str(row["id"]),
                    speaker=Speaker(id=str(row["speaker"])),
                    conversation_id=str(row["conversation_id"]),
                    reply_to=reply_to,
                    timestamp=row["timestamp"],
                    text=row["text"],
                    meta=metadata,
                )
            )

        # initializing corpus using utterance_list
        corpus = Corpus(utterances=utterance_list)
        if speakers_df is not None:
            corpus.update_metadata_from_df("speaker", speakers_df)
        if conversations_df is not None:
            corpus.update_metadata_from_df("conversation", conversations_df)

        return corpus
    
class MemStorageManager(StorageManager):
    """
    Concrete StorageManager implementation for in-memory data storage.
    Collections are implemented as vanilla Python dicts.
    """

    def __init__(self):
        super().__init__()

        # initialize component collections as dicts
        for key in self.data:
            self.data[key] = {}

    def get_collection_ids(self, component_type: str):
        return list(self.get_collection(component_type).keys())

    def has_data_for_component(self, component_type: str, component_id: str) -> bool:
        collection = self.get_collection(component_type)
        return component_id in collection

    def initialize_data_for_component(
        self, component_type: str, component_id: str, overwrite: bool = False, initial_value=None
    ):
        collection = self.get_collection(component_type)
        if overwrite or not self.has_data_for_component(component_type, component_id):
            collection[component_id] = initial_value if initial_value is not None else {}

    def get_data(
        self,
        component_type: str,
        component_id: str,
        property_name: Optional[str] = None,
        index=None,
    ):
        collection = self.get_collection(component_type)
        if component_id not in collection:
            raise KeyError(
                f"This StorageManager does not have an entry for the {component_type} with id {component_id}."
            )
        if property_name is None:
            return collection[component_id]
        else:
            return collection[component_id][property_name]

    def update_data(
        self,
        component_type: str,
        component_id: str,
        property_name: str,
        new_value,
        index=None,
    ):
        collection = self.get_collection(component_type)
        # don't create new collections if the ID is not found; this is supposed to be handled in the
        # CorpusComponent constructor so if the ID is missing that indicates something is wrong
        if component_id not in collection:
            raise KeyError(
                f"This StorageManager does not have an entry for the {component_type} with id {component_id}."
            )
        collection[component_id][property_name] = new_value

    def delete_data(
        self, component_type: str, component_id: str, property_name: Optional[str] = None
    ):
        collection = self.get_collection(component_type)
        if component_id not in collection:
            raise KeyError(
                f"This StorageManager does not have an entry for the {component_type} with id {component_id}."
            )
        if property_name is None:
            del collection[component_id]
        else:
            del collection[component_id][property_name]

    def clear_all_data(self):
        for key in self.data:
            self.data[key] = {}

    def count_entries(self, component_type: str):
        return len(self.get_collection(component_type))


class DBStorageManager(StorageManager):
    """
    Concrete StorageManager implementation for database-backed data storage.
    Collections are implemented as MongoDB collections.
    """

    def __init__(self, collection_prefix, db_host: Optional[str] = None):
        super().__init__()

        self.collection_prefix = collection_prefix
        self.client = MongoClient(db_host)
        self.db = self.client["convokit"]

        # this special lock is used for reconnecting to an existing DB, whereupon
        # it is known that all the data already exists and so the initialization
        # step can be skipped, greatly saving time
        self.bypass_init = False

        # initialize component collections as MongoDB collections in the convokit db
        for key in self.data:
            self.data[key] = self.db[self._get_collection_name(key)]

    def _get_collection_name(self, component_type: str) -> str:
        return f"{self.collection_prefix}_{component_type}"

    def get_collection_ids(self, component_type: str):
        return [
            doc["_id"]
            for doc in self.db[self._get_collection_name(component_type)].find(projection=["_id"])
        ]

    def has_data_for_component(self, component_type: str, component_id: str) -> bool:
        collection = self.get_collection(component_type)
        lookup = collection.find_one({"_id": component_id})
        return lookup is not None

    def initialize_data_for_component(
        self, component_type: str, component_id: str, overwrite: bool = False, initial_value=None
    ):
        if self.bypass_init:
            return
        collection = self.get_collection(component_type)
        if overwrite or not self.has_data_for_component(component_type, component_id):
            data = initial_value if initial_value is not None else {}
            collection.replace_one({"_id": component_id}, data, upsert=True)

    def get_data(
        self,
        component_type: str,
        component_id: str,
        property_name: Optional[str] = None,
        index=None,
    ):
        collection = self.get_collection(component_type)
        all_fields = collection.find_one({"_id": component_id})
        if all_fields is None:
            raise KeyError(
                f"This StorageManager does not have an entry for the {component_type} with id {component_id}."
            )
        if property_name is None:
            # if some data is known to be binary type, unpack it
            if index is not None:
                for key in all_fields:
                    if index.get(key, None) == ["bin"]:
                        all_fields[key] = pickle.loads(all_fields[key])
            # do not include the MongoDB-specific _id field
            del all_fields["_id"]
            return all_fields
        else:
            result = all_fields[property_name]
            if index is not None and index.get(property_name, None) == ["bin"]:
                # binary data must be unpacked
                result = pickle.loads(result)
            return result

    def update_data(
        self,
        component_type: str,
        component_id: str,
        property_name: str,
        new_value,
        index=None,
    ):
        data = self.get_data(component_type, component_id)
        if index is not None and index.get(property_name, None) == ["bin"]:
            # non-serializable types must go through pickling then be encoded as bson.Binary
            new_value = bson.Binary(pickle.dumps(new_value))
        data[property_name] = new_value
        collection = self.get_collection(component_type)
        collection.update_one({"_id": component_id}, {"$set": data})

    def delete_data(
        self, component_type: str, component_id: str, property_name: Optional[str] = None
    ):
        collection = self.get_collection(component_type)
        if property_name is None:
            # delete the entire document
            collection.delete_one({"_id": component_id})
        else:
            # delete only the specified property
            collection.update_one({"_id": component_id}, {"$unset": {property_name: ""}})

    def clear_all_data(self):
        for key in self.data:
            self.data[key].drop()
            self.data[key] = self.db[self._get_collection_name(key)]

    def count_entries(self, component_type: str):
        return self.get_collection(component_type).estimated_document_count()

class ConvoKitIndex:
    def __init__(
        self,
        owner,
        utterances_index: Optional[Dict[str, List[str]]] = None,
        speakers_index: Optional[Dict[str, List[str]]] = None,
        conversations_index: Optional[Dict[str, List[str]]] = None,
        overall_index: Optional[Dict[str, List[str]]] = None,
        vectors: Optional[List[str]] = None,
        version: Optional[int] = 0,
    ):
        self.owner = owner
        self.utterances_index = utterances_index if utterances_index is not None else {}
        self.speakers_index = speakers_index if speakers_index is not None else {}
        self.conversations_index = conversations_index if conversations_index is not None else {}
        self.overall_index = overall_index if overall_index is not None else {}
        self.indices = {
            "utterance": self.utterances_index,
            "conversation": self.conversations_index,
            "speaker": self.speakers_index,
            "corpus": self.overall_index,
        }
        self.vectors = set(vectors) if vectors is not None else set()
        self.version = version
        self.type_check = True  # toggle-able to enable/disable type checks on metadata additions
        self.lock_metadata_deletion = {"utterance": True, "conversation": True, "speaker": True}

    def create_new_index(self, obj_type: str, key: str):
        """
        Create a new entry in the obj_type index with a blank type list,
        representing an "Any" type which might be later refined.
        :param obj_type: utterance, conversation, or speaker
        :param key: string
        :param class_type: class type
        """
        if key not in self.indices[obj_type]:
            self.indices[obj_type][key] = []

    def update_index(self, obj_type: str, key: str, class_type: str):
        """
        Append the class_type to the index
        :param obj_type: utterance, conversation, or speaker
        :param key: string
        :param class_type: class type
        :return: None
        """
        assert type(key) == str
        assert "class" in class_type or class_type == "bin"
        if key not in self.indices[obj_type]:
            self.indices[obj_type][key] = []
        self.indices[obj_type][key].append(class_type)

    def set_index(self, obj_type: str, key: str, class_type: str):
        """
        Set the class_type of the index as [`class_type`].
        :param obj_type: utterance, conversation, or speaker
        :param key: string
        :param class_type: class type
        :return: None
        """
        assert type(key) == str
        assert "class" in class_type or class_type == "bin"
        self.indices[obj_type][key] = [class_type]

    def get_index(self, obj_type: str):
        return self.indices[obj_type]

    def del_from_index(self, obj_type: str, key: str):
        assert type(key) == str
        if key not in self.indices[obj_type]:
            return
        del self.indices[obj_type][key]
        #
        # corpus = self.owner
        # for corpus_obj in corpus.iter_objs(obj_type):
        #     if key in corpus_obj.meta:
        #         del corpus_obj.meta[key]

    def add_vector(self, vector_name):
        self.vectors.add(vector_name)

    def del_vector(self, vector_name):
        self.vectors.remove(vector_name)

    def update_from_dict(self, meta_index: Dict):
        self.conversations_index.update(meta_index["conversations-index"])
        self.utterances_index.update(meta_index["utterances-index"])
        speaker_index = "speakers-index" if "speakers-index" in meta_index else "users-index"
        self.speakers_index.update(meta_index[speaker_index])
        self.overall_index.update(meta_index["overall-index"])
        self.vectors = set(meta_index.get("vectors", set()))
        for index in self.indices.values():
            for k, v in index.items():
                if isinstance(v, str):
                    index[k] = [v]

        self.version = meta_index["version"]

    def to_dict(self, exclude_vectors: List[str] = None, force_version=None):
        retval = dict()
        retval["utterances-index"] = self.utterances_index
        retval["speakers-index"] = self.speakers_index
        retval["conversations-index"] = self.conversations_index
        retval["overall-index"] = self.overall_index

        if force_version is None:
            retval["version"] = self.version + 1
        else:
            retval["version"] = force_version

        if exclude_vectors is not None:
            retval["vectors"] = list(self.vectors - set(exclude_vectors))
        else:
            retval["vectors"] = list(self.vectors)

        return retval

    def enable_type_check(self):
        self.type_check = True

    def disable_type_check(self):
        self.type_check = False

    def __str__(self):
        return str(self.to_dict(force_version=self.version))

    def __repr__(self):
        return str(self)


### HELPER FUNCTIONS

def extract_meta_from_df(df):
    meta_cols = [col.split(".")[1] for col in df if col.startswith("meta")]
    return meta_cols

def get_corpus_dirpath(filename: str) -> Optional[str]:
    if filename is None:
        return None
    elif os.path.isdir(filename):
        return filename
    else:
        return os.path.dirname(filename)

def get_corpus_id(
    db_collection_prefix: Optional[str], filename: Optional[str], storage_type: str
) -> Optional[str]:
    if db_collection_prefix is not None:
        # treat the unique collection prefix as the ID (even if a filename is specified)
        corpus_id = db_collection_prefix
    elif filename is not None:
        # automatically derive an ID from the file path
        corpus_id = os.path.basename(os.path.normpath(filename))
    else:
        corpus_id = None

    if storage_type == "db" and corpus_id is not None:
        compatibility_msg = check_id_for_mongodb(corpus_id)
        if compatibility_msg is not None:
            random_id = create_safe_id()
            warn(
                f'Attempting to use "{corpus_id}" as DB collection prefix failed because: {compatibility_msg}. Will instead use randomly generated prefix {random_id}.'
            )
            corpus_id = random_id

    return corpus_id

def initialize_storage(
    corpus: "Corpus", storage: Optional[StorageManager], storage_type: str, db_host: Optional[str]
):
    if storage is not None:
        return storage
    else:
        if storage_type == "mem":
            return MemStorageManager()
        elif storage_type == "db":
            if db_host is None:
                db_host = corpus.config.db_host
            return DBStorageManager(corpus.id, db_host)
        else:
            raise ValueError(
                f"Unrecognized setting '{storage_type}' for storage type; should be either 'mem' or 'db'."
            )

def initialize_conversations(
    corpus, convos_data, convo_to_utts=None, fill_missing_convo_ids: bool = False
):
    """
    Initialize Conversation objects from utterances and conversations data.
    If a mapping from Conversation IDs to their constituent Utterance IDs is
    already known (e.g., as a side effect of a prior computation) they can be
    directly provided via the convo_to_utts parameter, otherwise the mapping
    will be computed by iteration over the Utterances in utt_dict.
    """
    if fill_missing_convo_ids:
        fill_missing_conversation_ids(corpus.utterances)

    # organize utterances by conversation
    if convo_to_utts is None:
        convo_to_utts = defaultdict(list)  # temp container identifying utterances by conversation
        for utt in corpus.utterances.values():
            convo_key = (
                utt.conversation_id
            )  # each conversation_id is considered a separate conversation
            convo_to_utts[convo_key].append(utt.id)
    conversations = {}
    for convo_id in convo_to_utts:
        # look up the metadata associated with this conversation, if any
        convo_data = convos_data.get(convo_id, None)
        if convo_data is not None:
            if KeyMeta in convo_data:
                convo_meta = convo_data[KeyMeta]
            else:
                convo_meta = convo_data
        else:
            convo_meta = None

        convo = Conversation(
            owner=corpus, id=convo_id, utterances=convo_to_utts[convo_id], meta=convo_meta
        )

        if convo_data is not None and KeyVectors in convo_data and KeyMeta in convo_data:
            convo.vectors = convo_data.get(KeyVectors, [])
        conversations[convo_id] = convo
    return conversations


def fill_missing_conversation_ids(utterances_dict: Dict[str, Utterance]) -> None:
    """
    Populates `conversation_id` in Utterances that have `conversation_id` set to `None`, with a Conversation root-specific generated ID
    :param utterances_dict:
    :return:
    """
    utts_without_convo_ids = [
        utt for utt in utterances_dict.values() if utt.conversation_id is None
    ]
    utt_ids_to_replier_ids = defaultdict(deque)
    convo_roots_without_convo_ids = []
    convo_roots_with_convo_ids = []
    for utt in utterances_dict.values():
        if utt.reply_to is None:
            if utt.conversation_id is None:
                convo_roots_without_convo_ids.append(utt.id)
            else:
                convo_roots_with_convo_ids.append(utt.id)
        else:
            utt_ids_to_replier_ids[utt.reply_to].append(utt.id)

    # connect the reply-to edges for convo roots without convo ids
    for root_utt_id in convo_roots_without_convo_ids:
        generated_conversation_id = Conversation.generate_default_conversation_id(
            utterance_id=root_utt_id
        )
        utterances_dict[root_utt_id].conversation_id = generated_conversation_id
        _update_reply_to_chain_with_conversation_id(
            utterances_dict=utterances_dict,
            utt_ids_to_replier_ids=utt_ids_to_replier_ids,
            root_utt_id=root_utt_id,
            conversation_id=generated_conversation_id,
        )

    # Previous section handles all *new* conversations
    # Next section handles utts that belong to existing conversations
    for root_utt_id in convo_roots_with_convo_ids:
        conversation_id = utterances_dict[root_utt_id].conversation_id
        _update_reply_to_chain_with_conversation_id(
            utterances_dict=utterances_dict,
            utt_ids_to_replier_ids=utt_ids_to_replier_ids,
            root_utt_id=root_utt_id,
            conversation_id=conversation_id,
        )

    # It's still possible to have utts that reply to non-existent utts
    # These are the utts that do not have a conversation_id even at this step
    for utt in utts_without_convo_ids:
        if utt.conversation_id is None:
            raise ValueError(
                f"Invalid Utterance found: Utterance {utt.id} replies to an Utterance '{utt.reply_to}' that does not exist."
            )

def _update_reply_to_chain_with_conversation_id(
    utterances_dict: Dict[str, Utterance],
    utt_ids_to_replier_ids: Dict[str, Iterable[str]],
    root_utt_id: str,
    conversation_id: str,
):
    repliers = utt_ids_to_replier_ids.get(root_utt_id, deque())
    while len(repliers) > 0:
        replier_id = repliers.popleft()
        utterances_dict[replier_id].conversation_id = conversation_id
        repliers.extend(utt_ids_to_replier_ids[replier_id])

def dump_corpus_component(
    corpus, dir_name, filename, obj_type, bin_name, exclude_vectors, fields_to_skip
):
    with open(os.path.join(dir_name, filename), "w") as f:
        d_bin = defaultdict(list)
        objs = defaultdict(dict)
        for obj_id in corpus.get_object_ids(obj_type):
            objs[obj_id][KeyMeta] = dump_helper_bin(
                corpus.get_object(obj_type, obj_id).meta, d_bin, fields_to_skip.get(obj_type, [])
            )
            obj_vectors = corpus.get_object(obj_type, obj_id).vectors
            objs[obj_id][KeyVectors] = (
                obj_vectors
                if exclude_vectors is None
                else list(set(obj_vectors) - set(exclude_vectors))
            )
        json.dump(objs, f)

        for name, l_bin in d_bin.items():
            with open(os.path.join(dir_name, name + "-{}-bin.p".format(bin_name)), "wb") as f_pk:
                pickle.dump(l_bin, f_pk)

BIN_DELIM_L, BIN_DELIM_R = "<##bin{", "}&&@**>"
KeyId = "id"
KeySpeaker = "speaker"
KeyConvoId = "conversation_id"
KeyReplyTo = "reply-to"
KeyTimestamp = "timestamp"
KeyText = "text"
DefinedKeys = {KeyId, KeySpeaker, KeyConvoId, KeyReplyTo, KeyTimestamp, KeyText}
KeyMeta = "meta"
KeyVectors = "vectors"

JSONLIST_BUFFER_SIZE = 1000

def dump_helper_bin(d: ConvoKitMeta, d_bin: Dict, fields_to_skip=None) -> Dict:  # object_idx
    """
    :param d: The ConvoKitMeta to encode
    :param d_bin: The dict of accumulated lists of binary attribs
    :return:
    """
    if fields_to_skip is None:
        fields_to_skip = []

    obj_idx = d.index.get_index(d.obj_type)
    d_out = {}
    for k, v in d.items():
        if k in fields_to_skip:
            continue
        try:
            if len(obj_idx[k]) > 0 and obj_idx[k][0] == "bin":
                d_out[k] = "{}{}{}".format(BIN_DELIM_L, len(d_bin[k]), BIN_DELIM_R)
                d_bin[k].append(v)
            else:
                d_out[k] = v
        except KeyError:
            # fails silently (object has no such metadata that was indicated in metadata index)
            pass
    return d_out

def dump_utterances(corpus, dir_name, exclude_vectors, fields_to_skip):
    with open(os.path.join(dir_name, "utterances.jsonl"), "w") as f:
        d_bin = defaultdict(list)

        for ut in corpus.iter_utterances():
            ut_obj = {
                KeyId: ut.id,
                KeyConvoId: ut.conversation_id,
                KeyText: ut.text,
                KeySpeaker: ut.speaker.id,
                KeyMeta: dump_helper_bin(ut.meta, d_bin, fields_to_skip.get("utterance", [])),
                KeyReplyTo: ut.reply_to,
                KeyTimestamp: ut.timestamp,
                KeyVectors: ut.vectors
                if exclude_vectors is None
                else list(set(ut.vectors) - set(exclude_vectors)),
            }
            json.dump(ut_obj, f)
            f.write("\n")

        for name, l_bin in d_bin.items():
            with open(os.path.join(dir_name, name + "-bin.p"), "wb") as f_pk:
                pickle.dump(l_bin, f_pk)

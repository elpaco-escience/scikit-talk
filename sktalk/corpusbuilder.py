#import random

import pandas as pd
import shutil
from typing import Collection, Callable, Set, Generator, Tuple, ValuesView, Union

from pandas import DataFrame
from tqdm import tqdm

try:
    from collections.abc import MutableMapping
except:
    from collections import MutableMapping
from numpy import isin
import json
from typing import Union

#from convokit.util import warn
#from .convoKitIndex import ConvoKitIndex
# from convokit.convokitConfig import ConvoKitConfig
# from convokit.util import create_safe_id
# from .convoKitMatrix import ConvoKitMatrix
# from .corpusUtil import *
# from .corpus_helpers import *
# from .storageManager import StorageManager

#DEPENDENCY CLASSES

from typing import List, Optional

# from convokit.util import warn
# from .convoKitMeta import ConvoKitMeta

import os
from typing import Optional
from yaml import load, Loader


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


_basic_types = {type(0), type(1.0), type("str"), type(True)}  # cannot include lists or dicts


def _optimized_type_check(val):
    # if type(obj)
    if type(val) in _basic_types:
        return str(type(val))
    else:
        try:
            json.dumps(val)
            return str(type(val))
        except (TypeError, OverflowError):
            return "bin"


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
        except AttributeError:  # for backwards compatibility when corpus objects are saved as binary data, e.g. wikiconv
            return "(" + str(copy) + ")"


from functools import total_ordering
from typing import Dict, List, Optional

# from .corpusComponent import CorpusComponent
# from .corpusUtil import *



from typing import Dict, Optional

# from convokit.util import warn
# from .corpusComponent import CorpusComponent
# from .speaker import Speaker


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
        speaker: Optional[str] = None,
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



@total_ordering
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

## HELPER FUNCTIONS
# FROM convokit.utils, corpushelper etc

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


# def dump_helper_bin(d: ConvoKitMeta, d_bin: Dict, fields_to_skip=None) -> Dict:  # object_idx
#     """
#     :param d: The ConvoKitMeta to encode
#     :param d_bin: The dict of accumulated lists of binary attribs
#     :return:
#     """
#     if fields_to_skip is None:
#         fields_to_skip = []

#     obj_idx = d.index.get_index(d.obj_type)
#     d_out = {}
#     for k, v in d.items():
#         if k in fields_to_skip:
#             continue
#         try:
#             if len(obj_idx[k]) > 0 and obj_idx[k][0] == "bin":
#                 d_out[k] = "{}{}{}".format(BIN_DELIM_L, len(d_bin[k]), BIN_DELIM_R)
#                 d_bin[k].append(v)
#             else:
#                 d_out[k] = v
#         except KeyError:
#             # fails silently (object has no such metadata that was indicated in metadata index)
#             pass
#     return d_out

def get_corpus_dirpath(filename: str) -> Optional[str]:
    if filename is None:
        return None
    elif os.path.isdir(filename):
        return filename
    else:
        return os.path.dirname(filename)

def get_utterance_ids(self) -> List[str]:
    """Produces a list of the unique IDs of all utterances in the
    Conversation, which can be used in calls to get_utterance() to retrieve
    specific utterances. Provides no ordering guarantees for the list.
    :return: a list of IDs of Utterances in the Conversation
    """
    # pass a copy of the list
    return self._utterance_ids[:]

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


def extract_meta_from_df(df):
    meta_cols = [col.split(".")[1] for col in df if col.startswith("meta")]
    return meta_cols

def update_metadata_from_df(self, obj_type, df):
    assert obj_type in ["utterance", "speaker", "conversation"]
    meta_cols = extract_meta_from_df(df)
    df.columns = [col.replace("meta.", "") for col in df.columns]
    df = df.set_index("id")
    for obj in self.iter_objs(obj_type):
        obj_meta = df.loc[obj.id][meta_cols].to_dict() if meta_cols else None
        if obj_meta is not None:
            obj.meta.update(obj_meta)
    return self



### BEGIN CORPUS CLASS


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
        #storage: Optional[StorageManager] = None,
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

    @classmethod
    def reconnect_to_db(cls, db_collection_prefix: str, db_host: Optional[str] = None):
        """
        Factory method for a Corpus instance backed by an already-existing database (e.g.,
        one that was created in a previous run of a Python script or interactive session).
        This can be used to reconnect to existing Corpus data that you still want to use
        without having to reload the data from the source file; this can happen for example
        if your script crashed in the middle of working with the Corpus and you want to
        resume where you left off.
        """
        # create a blank Corpus that will hold the data
        result = cls(db_collection_prefix=db_collection_prefix, db_host=db_host, storage_type="db")
        # through the constructor, the blank Corpus' StorageManager is now connected
        # to the DB. Next use the DB contents to populate the corpus components.
        init_corpus_from_storage_manager(result)

        return result

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

    # def get_utterance(self, utt_id: str) -> Utterance:
    #     """
    #     Gets Utterance of the specified id from the corpus
    #     :param utt_id: id of Utterance
    #     :return: Utterance
    #     """
    #     return self.utterances[utt_id]

    # def get_conversation(self, convo_id: str) -> Conversation:
    #     """
    #     Gets Conversation of the specified id from the corpus
    #     :param convo_id: id of Conversation
    #     :return: Conversation
    #     """
    #     return self.conversations[convo_id]

    # def get_speaker(self, speaker_id: str) -> Speaker:
    #     """
    #     Gets Speaker of the specified id from the corpus
    #     :param speaker_id: id of Speaker
    #     :return: Speaker
    #     """
    #     return self.speakers[speaker_id]

    # def get_object(self, obj_type: str, oid: str):
    #     """
    #     General Corpus object getter. Gets Speaker / Utterance / Conversation of specified id from the Corpus
    #     :param obj_type: "speaker", "utterance", or "conversation"
    #     :param oid: object id
    #     :return: Corpus object of specified object type with specified object id
    #     """
    #     assert obj_type in ["speaker", "utterance", "conversation"]
    #     if obj_type == "speaker":
    #         return self.get_speaker(oid)
    #     elif obj_type == "utterance":
    #         return self.get_utterance(oid)
    #     else:
    #         return self.get_conversation(oid)

    # def has_utterance(self, utt_id: str) -> bool:
    #     """
    #     Checks if an Utterance of the specified id exists in the Corpus
    #     :param utt_id: id of Utterance
    #     :return: True if Utterance of specified id is present, False otherwise
    #     """
    #     return utt_id in self.utterances

    # def has_conversation(self, convo_id: str) -> bool:
    #     """
    #     Checks if a Conversation of the specified id exists in the Corpus
    #     :param convo_id: id of Conversation
    #     :return: True if Conversation of specified id is present, False otherwise
    #     """
    #     return convo_id in self.conversations

    # def has_speaker(self, speaker_id: str) -> bool:
    #     """
    #     Checks if a Speaker of the specified id exists in the Corpus
    #     :param speaker_id: id of Speaker
    #     :return: True if Speaker of specified id is present, False otherwise
    #     """
    #     return speaker_id in self.speakers

    # def random_utterance(self) -> Utterance:
    #     """
    #     Get a random Utterance from the Corpus
    #     :return: a random Utterance
    #     """
    #     return random.choice(list(self.utterances.values()))

    # def random_conversation(self) -> Conversation:
    #     """
    #     Get a random Conversation from the Corpus
    #     :return: a random Conversation
    #     """
    #     return random.choice(list(self.conversations.values()))

    # def random_speaker(self) -> Speaker:
    #     """
    #     Get a random Speaker from the Corpus
    #     :return: a random Speaker
    #     """
    #     return random.choice(list(self.speakers.values()))

    # def iter_utterances(
    #     self, selector: Optional[Callable[[Utterance], bool]] = lambda utt: True
    # ) -> Generator[Utterance, None, None]:
    #     """
    #     Get utterances in the Corpus, with an optional selector that filters for Utterances that should be included.
    #     :param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude).
    #         By default, the selector includes all Utterances in the Corpus.
    #     :return: a generator of Utterances
    #     """
    #     for v in self.utterances.values():
    #         if selector(v):
    #             yield v

    # def get_utterances_dataframe(
    #     self,
    #     selector: Optional[Callable[[Utterance], bool]] = lambda utt: True,
    #     exclude_meta: bool = False,
    # ):
    #     """
    #     Get a DataFrame of the utterances with fields and metadata attributes, with an optional selector that filters
    #     utterances that should be included. Edits to the DataFrame do not change the corpus in any way.
    #     :param exclude_meta: whether to exclude metadata
    #     :param selector: a (lambda) function that takes a Utterance and returns True or False (i.e. include / exclude).
    #         By default, the selector includes all Utterances in the Corpus.
    #     :return: a pandas DataFrame
    #     """
    #     return get_utterances_dataframe(self, selector, exclude_meta)

    # def iter_conversations(
    #     self, selector: Optional[Callable[[Conversation], bool]] = lambda convo: True
    # ) -> Generator[Conversation, None, None]:
    #     """
    #     Get conversations in the Corpus, with an optional selector that filters for Conversations that should be included
    #     :param selector: a (lambda) function that takes a Conversation and returns True or False (i.e. include / exclude).
    #         By default, the selector includes all Conversations in the Corpus.
    #     :return: a generator of Conversations
    #     """
    #     for v in self.conversations.values():
    #         if selector(v):
    #             yield v

    # def get_conversations_dataframe(
    #     self,
    #     selector: Optional[Callable[[Conversation], bool]] = lambda convo: True,
    #     exclude_meta: bool = False,
    # ):
    #     """
    #     Get a DataFrame of the conversations with fields and metadata attributes, with an optional selector that filters
    #     for conversations that should be included. Edits to the DataFrame do not change the corpus in any way.
    #     :param exclude_meta: whether to exclude metadata
    #     :param selector: a (lambda) function that takes a Conversation and returns True or False (i.e. include / exclude).
    #         By default, the selector includes all Conversations in the Corpus.
    #     :return: a pandas DataFrame
    #     """
    #     return get_conversations_dataframe(self, selector, exclude_meta)

    # def iter_speakers(
    #     self, selector: Optional[Callable[[Speaker], bool]] = lambda speaker: True
    # ) -> Generator[Speaker, None, None]:
    #     """
    #     Get Speakers in the Corpus, with an optional selector that filters for Speakers that should be included
    #     :param selector: a (lambda) function that takes a Speaker and returns True or False (i.e. include / exclude).
    #         By default, the selector includes all Speakers in the Corpus.
    #     :return: a generator of Speakers
    #     """

    #     for speaker in self.speakers.values():
    #         if selector(speaker):
    #             yield speaker

    # def get_speakers_dataframe(
    #     self,
    #     selector: Optional[Callable[[Speaker], bool]] = lambda utt: True,
    #     exclude_meta: bool = False,
    # ):
    #     """
    #     Get a DataFrame of the Speakers with fields and metadata attributes, with an optional selector that filters
    #     Speakers that should be included. Edits to the DataFrame do not change the corpus in any way.
    #     :param exclude_meta: whether to exclude metadata
    #     :param selector: selector: a (lambda) function that takes a Speaker and returns True or False
    #         (i.e. include / exclude). By default, the selector includes all Speakers in the Corpus.
    #     :return: a pandas DataFrame
    #     """
    #     return get_speakers_dataframe(self, selector, exclude_meta)

    # def iter_objs(
    #     self,
    #     obj_type: str,
    #     selector: Callable[[Union[Speaker, Utterance, Conversation]], bool] = lambda obj: True,
    # ):
    #     """
    #     Get Corpus objects of specified type from the Corpus, with an optional selector that filters for Corpus object that should be included
    #     :param obj_type: "speaker", "utterance", or "conversation"
    #     :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude).
    #         By default, the selector includes all objects of the specified type in the Corpus.
    #     :return: a generator of Speakers
    #     """

    #     assert obj_type in ["speaker", "utterance", "conversation"]
    #     obj_iters = {
    #         "conversation": self.iter_conversations,
    #         "speaker": self.iter_speakers,
    #         "utterance": self.iter_utterances,
    #     }

    #     return obj_iters[obj_type](selector)

    # def get_utterance_ids(
    #     self, selector: Optional[Callable[[Utterance], bool]] = lambda utt: True
    # ) -> List[str]:
    #     """
    #     Get a list of ids of Utterances in the Corpus, with an optional selector that filters for Utterances that should be included
    #     :param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude).
    #         By default, the selector includes all Utterances in the Corpus.
    #     :return: list of Utterance ids
    #     """
    #     return [utt.id for utt in self.iter_utterances(selector)]

    # def get_conversation_ids(
    #     self, selector: Optional[Callable[[Conversation], bool]] = lambda convo: True
    # ) -> List[str]:
    #     """
    #     Get a list of ids of Conversations in the Corpus, with an optional selector that filters for Conversations that should be included
    #     :param selector: a (lambda) function that takes a Conversation and returns True or False (i.e. include / exclude).
    #         By default, the selector includes all Conversations in the Corpus.
    #     :return: list of Conversation ids
    #     """
    #     return [convo.id for convo in self.iter_conversations(selector)]

    # def get_speaker_ids(
    #     self, selector: Optional[Callable[[Speaker], bool]] = lambda speaker: True
    # ) -> List[str]:
    #     """
    #     Get a list of ids of Speakers in the Corpus, with an optional selector that filters for Speakers that should be included
    #     :param selector: a (lambda) function that takes a Speaker and returns True or False (i.e. include / exclude).
    #         By default, the selector includes all Speakers in the Corpus.
    #     :return: list of Speaker ids
    #     """
    #     return [speaker.id for speaker in self.iter_speakers(selector)]

    # def get_object_ids(
    #     self,
    #     obj_type: str,
    #     selector: Callable[[Union[Speaker, Utterance, Conversation]], bool] = lambda obj: True,
    # ):
    #     """
    #     Get a list of ids of Corpus objects of the specified type in the Corpus, with an optional selector that filters for objects that should be included
    #     :param obj_type: "speaker", "utterance", or "conversation"
    #     :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude).
    #         By default, the selector includes all objects of the specified type in the Corpus.
    #     :return: list of Corpus object ids
    #     """
    #     assert obj_type in ["speaker", "utterance", "conversation"]
    #     return [obj.id for obj in self.iter_objs(obj_type, selector)]

    # def filter_conversations_by(self, selector: Callable[[Conversation], bool]):
    #     """
    #     Mutate the corpus by filtering for a subset of Conversations within the Corpus.
    #     :param selector: function for selecting which Conversations to keep
    #     :return: the mutated Corpus
    #     """

    #     self.conversations = {
    #         convo_id: convo for convo_id, convo in self.conversations.items() if selector(convo)
    #     }
    #     utt_ids = set(
    #         [utt for convo in self.conversations.values() for utt in convo.get_utterance_ids()]
    #     )
    #     self.utterances = {utt.id: utt for utt in self.utterances.values() if utt.id in utt_ids}
    #     speaker_ids = set([utt.speaker.id for utt in self.utterances.values()])
    #     self.speakers = {
    #         speaker.id: speaker for speaker in self.speakers.values() if speaker.id in speaker_ids
    #     }
    #     self.update_speakers_data()
    #     self.reinitialize_index()

    #     # clear all storage entries corresponding to filtered-out components
    #     meta_ids = [self.meta.storage_key]
    #     for utt in self.iter_utterances():
    #         meta_ids.append(utt.meta.storage_key)
    #     for convo in self.iter_conversations():
    #         meta_ids.append(convo.meta.storage_key)
    #     for speaker in self.iter_speakers():
    #         meta_ids.append(speaker.meta.storage_key)
    #     self.storage.purge_obsolete_entries(
    #         self.get_utterance_ids(), self.get_conversation_ids(), self.get_speaker_ids(), meta_ids
    #     )

    #     return self

    # @staticmethod
    # def filter_utterances(source_corpus: "Corpus", selector: Callable[[Utterance], bool]):
    #     """
    #     Returns a new corpus that includes only a subset of Utterances from the source Corpus. This filtering provides no
    #     guarantees with regard to maintaining conversational integrity and should be used with care.
    #     Vectors are not preserved. The source corpus will be invalidated and will no longer be usable.
    #     :param source_corpus: the Corpus to subset from
    #     :param selector: function for selecting which
    #     :return: a new Corpus with a subset of the Utterances
    #     """
    #     utts = list(source_corpus.iter_utterances(selector))
    #     new_corpus = Corpus(utterances=utts)
    #     for convo in new_corpus.iter_conversations():
    #         convo.meta.update(source_corpus.get_conversation(convo.id).meta)

    #     # original Corpus is invalidated and no longer usable; clear all data from
    #     # its now-orphaned StorageManager to avoid having duplicates in memory
    #     source_corpus.storage.clear_all_data()

    #     return new_corpus

    # @staticmethod
    # def reindex_conversations(
    #     source_corpus: "Corpus",
    #     new_convo_roots: List[str],
    #     preserve_corpus_meta: bool = True,
    #     preserve_convo_meta: bool = True,
    #     verbose=True,
    # ) -> "Corpus":
    #     """
    #     Generates a new Corpus from source Corpus with specified list of utterance ids to use as conversation ids.
    #     The subtrees denoted by these utterance ids should be distinct and should not overlap, otherwise there may be unexpected behavior.
    #     Vectors are not preserved. The source Corpus will be invalidated and no longer usable.
    #     :param source_corpus: the Corpus containing the original data to select from
    #     :param new_convo_roots: List of utterance ids to use as conversation ids
    #     :param preserve_corpus_meta: set as True to copy original Corpus metadata to new Corpus
    #     :param preserve_convo_meta: set as True to copy original Conversation metadata to new Conversation metadata
    #         (For each new conversation, use the metadata of the conversation that the utterance belonged to.)
    #     :param verbose: whether to print a warning when
    #     :return: new Corpus with reindexed Conversations
    #     """ ""
    #     new_convo_roots = set(new_convo_roots)
    #     for convo in source_corpus.iter_conversations():
    #         try:
    #             convo.initialize_tree_structure()
    #         except ValueError as e:
    #             if verbose:
    #                 warn(str(e))

    #     new_corpus_utts = []
    #     original_utt_to_convo_id = dict()

    #     for utt_id in new_convo_roots:
    #         orig_convo = source_corpus.get_conversation(
    #             source_corpus.get_utterance(utt_id).conversation_id
    #         )
    #         original_utt_to_convo_id[utt_id] = orig_convo.id
    #         try:
    #             subtree = orig_convo.get_subtree(utt_id)
    #             new_root_utt = subtree.utt
    #             new_root_utt.reply_to = None
    #             subtree_utts = [node.utt for node in subtree.bfs_traversal()]
    #             for utt in subtree_utts:
    #                 utt.conversation_id = utt_id
    #             new_corpus_utts.extend(subtree_utts)
    #         except ValueError:
    #             continue

    #     new_corpus = Corpus(utterances=new_corpus_utts)

    #     if preserve_corpus_meta:
    #         new_corpus.meta.update(source_corpus.meta)

    #     if preserve_convo_meta:
    #         for convo in new_corpus.iter_conversations():
    #             convo.meta["original_convo_meta"] = source_corpus.get_conversation(
    #                 original_utt_to_convo_id[convo.id]
    #             ).meta.to_dict()
    #             convo.meta["original_convo_id"] = original_utt_to_convo_id[convo.id]
    #     if verbose:
    #         missing_convo_roots = list(
    #             set(new_convo_roots) - set(new_corpus.get_conversation_ids())
    #         )
    #         if len(missing_convo_roots) > 0:
    #             warn("Failed to find some of the specified new convo roots:\n")
    #             print(missing_convo_roots)

    #     # original Corpus is invalidated and no longer usable; clear all data from
    #     # its now-orphaned StorageManager to avoid having duplicates in memory
    #     source_corpus.storage.clear_all_data()

    #     return new_corpus

    # def get_meta(self) -> Dict:
    #     return self.meta

    # def add_meta(self, key: str, value) -> None:
    #     self.meta[key] = value

    # def speaking_pairs(
    #     self,
    #     selector: Optional[Callable[[Speaker, Speaker], bool]] = lambda speaker1, speaker2: True,
    #     speaker_ids_only: bool = False,
    # ) -> Set[Tuple[str, str]]:
    #     """
    #     Get all directed speaking pairs (a, b) of speakers such that a replies to b at least once in the dataset.
    #     :param selector: optional function that takes in a Speaker and a replied-to Speaker and returns True to include
    #         the pair in the result, or False otherwise.
    #     :param speaker_ids_only: if True, return just pairs of speaker names rather than speaker objects.
    #     :type speaker_ids_only: bool
    #     :return: Set containing all speaking pairs selected by the selector function, or all speaking pairs in the
    #         dataset if no selector function was used.
    #     """
    #     pairs = set()
    #     for utt2 in self.iter_utterances():
    #         if (
    #             utt2.speaker is not None
    #             and utt2.reply_to is not None
    #             and self.has_utterance(utt2.reply_to)
    #         ):
    #             utt1 = self.get_utterance(utt2.reply_to)
    #             if utt1.speaker is not None:
    #                 if selector(utt2.speaker, utt1.speaker):
    #                     pairs.add(
    #                         (utt2.speaker.id, utt1.speaker.id)
    #                         if speaker_ids_only
    #                         else (utt2.speaker, utt1.speaker)
    #                     )
    #     return pairs

    # def directed_pairwise_exchanges(
    #     self,
    #     selector: Optional[Callable[[Speaker, Speaker], bool]] = lambda speaker1, speaker2: True,
    #     speaker_ids_only: bool = False,
    # ) -> Dict[Tuple, List[Utterance]]:
    #     """
    #     Get all directed pairwise exchanges in the dataset.
    #     :param selector: optional function that takes in a speaking speaker and a replied-to speaker and
    #         returns True to include the pair in the result, or False otherwise.
    #     :param speaker_ids_only: if True, index conversations
    #         by speaker ids rather than Speaker objects.
    #     :type speaker_ids_only: bool
    #     :return: Dictionary mapping (speaker, target) tuples to a list of
    #         utterances given by the speaker in reply to the target.
    #     """
    #     pairs = defaultdict(list)
    #     for u2 in self.iter_utterances():
    #         if u2.speaker is not None and u2.reply_to is not None:
    #             u1 = self.get_utterance(u2.reply_to)
    #             if u1.speaker is not None:
    #                 if selector(u2.speaker, u1.speaker):
    #                     key = (
    #                         (u2.speaker.id, u1.speaker.id)
    #                         if speaker_ids_only
    #                         else (u2.speaker, u1.speaker)
    #                     )
    #                     pairs[key].append(u2)

    #     return pairs

    # @staticmethod
    # def _merge_utterances(
    #     utts1: List[Utterance], utts2: List[Utterance], warnings: bool
    # ) -> ValuesView[Utterance]:
    #     """
    #     Helper function for merge().
    #     Combine two collections of utterances into a single dictionary of Utterance id -> Utterance.
    #     If metadata of utterances in the two collections share the same key, but different values,
    #     the second collections' utterance metadata will be used.
    #     May mutate original collections of Utterances.
    #     Prints warnings when:
    #     1) Utterances with same id from this and other collection do not share the same data
    #     2) Utterance metadata has different values for the same key, and overwriting occurs
    #     :param utts1: First collection of Utterances
    #     :param utts2: Second collection of Utterances
    #     :param warnings: whether to print warnings when conflicting data is found.
    #     :return: ValuesView for merged set of utterances
    #     """
    #     seen_utts = dict()

    #     # Merge UTTERANCE metadata
    #     # Add all the utterances from this corpus
    #     for utt in utts1:
    #         seen_utts[utt.id] = utt

    #     # Add all the utterances from the other corpus, checking for data sameness and updating metadata as appropriate
    #     for utt in utts2:
    #         if utt.id in seen_utts:
    #             prev_utt = seen_utts[utt.id]
    #             if prev_utt == utt:
    #                 # other utterance metadata is ignored if data is not matched
    #                 for key, val in utt.meta.items():
    #                     if key in prev_utt.meta and prev_utt.meta[key] != val:
    #                         if warnings:
    #                             warn(
    #                                 "Found conflicting values for Utterance {} for metadata key: {}. "
    #                                 "Overwriting with other corpus's Utterance metadata.".format(
    #                                     repr(utt.id), repr(key)
    #                                 )
    #                             )
    #                     prev_utt.meta[key] = val
    #             else:
    #                 if warnings:
    #                     warn(
    #                         "Utterances with same id do not share the same data:\n"
    #                         + str(prev_utt)
    #                         + "\n"
    #                         + str(utt)
    #                         + "\n"
    #                         + "Ignoring second corpus's utterance."
    #                     )
    #         else:
    #             seen_utts[utt.id] = utt

    #     return seen_utts.values()

    # @staticmethod
    # def _collect_speaker_data(
    #     utt_sets: Collection[Collection[Utterance]],
    # ) -> Tuple[Dict[str, Speaker], Dict[str, Dict[str, str]], Dict[str, Dict[str, bool]]]:
    #     """
    #     Helper function for merge().
    #     Iterates through the input set of utterances, to collect Speaker data and metadata.
    #     Collect Speaker metadata in another Dictionary indexed by Speaker ID
    #     Track if conflicting speaker metadata is found in another dictionary
    #     :param utt_sets: Collections of collections of Utterances to extract Speakers from
    #     :return: speaker metadata and the corresponding tracker
    #     """
    #     # Collect SPEAKER data and metadata
    #     speakers_data = {}
    #     speakers_meta = defaultdict(lambda: defaultdict(str))
    #     speakers_meta_conflict = defaultdict(lambda: defaultdict(bool))
    #     for utt_set in utt_sets:
    #         for utt in utt_set:
    #             if utt.speaker.id not in speakers_data:
    #                 speakers_data[utt.speaker.id] = utt.speaker
    #             for meta_key, meta_val in utt.speaker.meta.items():
    #                 curr = speakers_meta[utt.speaker][meta_key]
    #                 if curr != meta_val:
    #                     if curr != "":
    #                         speakers_meta_conflict[utt.speaker][meta_key] = True
    #                     speakers_meta[utt.speaker][meta_key] = meta_val

    #     return speakers_data, speakers_meta, speakers_meta_conflict

    # @staticmethod
    # def _update_corpus_speaker_data(
    #     new_corpus, speakers_meta: Dict, speakers_meta_conflict: Dict, warnings: bool
    # ) -> None:
    #     """
    #     Helper function for merge().
    #     Update new_corpus's Speakers' data (utterance and conversation lists) and metadata
    #     Prints a warning if multiple values are found for any speaker's metadata key; latest speaker metadata is used
    #     :param speakers_meta: Dictionary indexed by Speaker ID, containing the collected Speaker metadata
    #     :param speakers_meta_conflict: Dictionary indexed by Speaker ID, indicating if there were value conflicts for the associated meta keys
    #     :return: None (mutates the new_corpus's Speakers)
    #     """
    #     # Update SPEAKER data and metadata with merged versions
    #     for speaker in new_corpus.iter_speakers():
    #         for meta_key, meta_val in speakers_meta[speaker].items():
    #             if speakers_meta_conflict[speaker][meta_key]:
    #                 if warnings:
    #                     warn(
    #                         "Multiple values found for {} for metadata key: {}. "
    #                         "Taking the latest one found".format(speaker, repr(meta_key))
    #                     )
    #             speaker.meta[meta_key] = meta_val

    # def _reinitialize_index_helper(self, new_index, obj_type):
    #     """
    #     Helper for reinitializing the index of the different Corpus object types
    #     :param new_index: new ConvoKitIndex object
    #     :param obj_type: utterance, speaker, or conversation
    #     :return: None (mutates new_index)
    #     """
    #     for obj in self.iter_objs(obj_type):
    #         for key, value in obj.meta.items():
    #             ConvoKitMeta._check_type_and_update_index(new_index, obj_type, key, value)
    #         obj.meta.index = new_index

    # def reinitialize_index(self):
    #     """
    #     Reinitialize the Corpus Index from scratch.
    #     :return: None (sets the .meta_index of Corpus and of the corpus component objects)
    #     """
    #     old_index = self.meta_index
    #     new_index = ConvoKitIndex(self)

    #     self._reinitialize_index_helper(new_index, "utterance")
    #     self._reinitialize_index_helper(new_index, "speaker")
    #     self._reinitialize_index_helper(new_index, "conversation")

    #     for key, value in self.meta.items():  # overall
    #         new_index.update_index("corpus", key, str(type(value)))

    #     new_index.version = old_index.version
    #     self.meta_index = new_index

    # @staticmethod
    # def merge(primary: "Corpus", secondary: "Corpus", warnings: bool = True):
    #     """
    #     Merges two corpora (one primary and one secondary), creating a new Corpus with their combined data.
    #     Utterances with the same id must share the same data. In case of conflicts,
    #     the primary Corpus will take precedence and the conflicting Utterance from secondary
    #     will be ignored. A warning is printed when this happens.
    #     If metadata of the primary Corpus (or its conversations / utterances) shares a key with the metadata of the
    #     secondary Corpus, the secondary's metadata (or its conversations / utterances) values will be used. A warning
    #     is printed when this happens.
    #     Will invalidate primary and secondary in the process.
    #     The resulting Corpus will inherit the primary Corpus's id and version number.
    #     :param primary: the primary Corpus
    #     :param secondary: the secondary Corpus
    #     :param warnings: print warnings when data conflicts are encountered
    #     :return: new Corpus constructed from combined lists of utterances
    #     """
    #     utts1 = list(primary.iter_utterances())
    #     utts2 = list(secondary.iter_utterances())
    #     combined_utts = list(primary._merge_utterances(utts1, utts2, warnings=warnings))
    #     # Note that we collect Speakers from the utt sets directly instead of the combined utts, otherwise
    #     # differences in Speaker meta will not be registered for duplicate Utterances (because utts would be discarded
    #     # during merging)
    #     combined_speakers, speakers_meta, speakers_meta_conflict = primary._collect_speaker_data(
    #         [utts1, utts2]
    #     )
    #     # Ensure that all attributions of an Utterance to the same speaker ID actually
    #     # map to the same Speaker instance. Otherwise, you can end up with two
    #     # Utterances that appear to have the same author but actually point to two
    #     # identical-but-distinct Speaker objects, which can result in unintuivie
    #     # behavior down the line.
    #     for utt in combined_utts:
    #         intended_speaker = combined_speakers[utt.speaker.id]
    #         if not (utt.speaker is intended_speaker):
    #             utt.speaker = intended_speaker
    #     new_corpus = Corpus(utterances=combined_utts)
    #     Corpus._update_corpus_speaker_data(
    #         new_corpus, speakers_meta, speakers_meta_conflict, warnings=warnings
    #     )

    #     # Merge CORPUS metadata
    #     new_corpus.meta.reinitialize_from(primary.meta)
    #     for key, val in secondary.meta.items():
    #         if key in new_corpus.meta and new_corpus.meta[key] != val:
    #             if warnings:
    #                 warn(
    #                     "Found conflicting values for primary Corpus metadata key: {}. "
    #                     "Overwriting with secondary Corpus's metadata.".format(repr(key))
    #                 )
    #         new_corpus.meta[key] = val

    #     # Merge CONVERSATION metadata
    #     convos1 = primary.iter_conversations()
    #     convos2 = secondary.iter_conversations()

    #     for convo in convos1:
    #         new_corpus.get_conversation(convo.id).meta.reinitialize_from(convo.meta)

    #     for convo in convos2:
    #         for key, val in convo.meta.items():
    #             curr_meta = new_corpus.get_conversation(convo.id).meta
    #             if key in curr_meta and curr_meta[key] != val:
    #                 if warnings:
    #                     warn(
    #                         "Found conflicting values for Conversation {} for metadata key: {}. "
    #                         "Overwriting with secondary corpus's Conversation metadata.".format(
    #                             repr(convo.id), repr(key)
    #                         )
    #                     )
    #             curr_meta[key] = val

    #     new_corpus.update_speakers_data()
    #     new_corpus.reinitialize_index()

    #     # source corpora are now invalidated and all needed data has been copied
    #     # into the new merged corpus; clear the source corpora's storage to
    #     # prevent having duplicates in memory
    #     primary.storage.clear_all_data()
    #     secondary.storage.clear_all_data()

    #     return new_corpus

    # def add_utterances(self, utterances=List[Utterance], warnings: bool = False, with_checks=True):
    #     """
    #     Add utterances to the Corpus.
    #     If the corpus has utterances that share an id with an utterance in the input utterance list,
    #     Optional warnings will be printed:
    #     - if the utterances with same id do not share the same data (added utterance is ignored)
    #     - added utterances' metadata have the same key but different values (added utterance's metadata will overwrite)
    #     :param utterances: Utterances to be added to the Corpus
    #     :param warnings: set to True for warnings to be printed
    #     :param with_checks: set to True if checks on utterance and metadata overlaps are desired. Set to False if newly added utterances are guaranteed to be new and share the same set of metadata keys.
    #     :return: a Corpus with the utterances from this Corpus and the input utterances combined
    #     """
    #     if with_checks:
    #         # leverage the merge method's _merge_utterances method to run the checks
    #         # (but then run a subsequent filtering operation since we aren't actually doing a merge)
    #         added_utt_ids = {utt.id for utt in utterances}
    #         combined_utts = self._merge_utterances(
    #             list(self.iter_utterances()), utterances, warnings
    #         )
    #         combined_speakers, speakers_meta, speakers_meta_conflict = self._collect_speaker_data(
    #             [list(self.iter_utterances()), utterances]
    #         )
    #         utterances = [utt for utt in combined_utts if utt.id in added_utt_ids]
    #         for utt in utterances:
    #             intended_speaker = combined_speakers[utt.speaker.id]
    #             if not (utt.speaker is intended_speaker):
    #                 utt.speaker = intended_speaker

    #     new_speakers = {u.speaker.id: u.speaker for u in utterances}
    #     new_utterances = {u.id: u for u in utterances}
    #     for speaker in new_speakers.values():
    #         speaker.owner = self
    #     for utt in new_utterances.values():
    #         utt.owner = self

    #     # update corpus speakers
    #     for new_speaker_id, new_speaker in new_speakers.items():
    #         if new_speaker_id not in self.speakers:
    #             self.speakers[new_speaker_id] = new_speaker

    #     # update corpus utterances + (link speaker -> utt)
    #     for new_utt_id, new_utt in new_utterances.items():
    #         if new_utt_id not in self.utterances:
    #             self.utterances[new_utt_id] = new_utt
    #             self.speakers[new_utt.speaker.id]._add_utterance(new_utt)

    #     # add convo ids if new utts are missing convo ids
    #     fill_missing_conversation_ids(self.utterances)

    #     # update corpus conversations + (link convo <-> utt)
    #     new_convos = defaultdict(list)
    #     convo_id_to_root_utt_id = dict()
    #     for utt in new_utterances.values():
    #         if utt.conversation_id in self.conversations:
    #             if (not with_checks) or (
    #                 utt.id not in self.conversations[utt.conversation_id].get_utterance_ids()
    #             ):
    #                 self.conversations[utt.conversation_id]._add_utterance(utt)
    #         else:
    #             new_convos[utt.conversation_id].append(utt.id)
    #         if utt.reply_to is None:
    #             convo_id_to_root_utt_id[utt.conversation_id] = utt.id

    #     for convo_id, convo_utts in new_convos.items():
    #         new_convo = Conversation(owner=self, id=convo_id, utterances=convo_utts, meta=None)
    #         self.conversations[convo_id] = new_convo
    #         # (link speaker -> convo)
    #         convo_root_utt_id = convo_id_to_root_utt_id[convo_id]
    #         convo_root_utt = new_convo.get_utterance(convo_root_utt_id)
    #         new_convo_speaker = self.speakers[convo_root_utt.speaker.id]
    #         new_convo_speaker._add_conversation(new_convo)

    #     # update speaker metadata (only in cases of conflict)
    #     if with_checks:
    #         Corpus._update_corpus_speaker_data(
    #             self, speakers_meta, speakers_meta_conflict, warnings
    #         )

    #     return self

    # def update_speakers_data(self) -> None:
    #     """
    #     Updates the conversation and utterance lists of every Speaker in the Corpus
    #     :return: None
    #     """
    #     speakers_utts = defaultdict(list)
    #     speakers_convos = defaultdict(list)

    #     for utt in self.iter_utterances():
    #         speakers_utts[utt.speaker.id].append(utt)

    #     for convo in self.iter_conversations():
    #         for utt in convo.iter_utterances():
    #             speakers_convos[utt.speaker.id].append(convo)

    #     for speaker in self.iter_speakers():
    #         speaker.utterances = {utt.id: utt for utt in speakers_utts[speaker.id]}
    #         speaker.conversations = {convo.id: convo for convo in speakers_convos[speaker.id]}

    # def print_summary_stats(self) -> None:
    #     """
    #     Helper function for printing the number of Speakers, Utterances, and Conversations in this Corpus
    #     :return: None
    #     """
    #     print("Number of Speakers: {}".format(len(self.speakers)))
    #     print("Number of Utterances: {}".format(len(self.utterances)))
    #     print("Number of Conversations: {}".format(len(self.conversations)))

    # def delete_metadata(self, obj_type: str, attribute: str):
    #     """
    #     Delete a specified metadata attribute from all Corpus components of the specified object type.
    #     Note that cancelling this method before it runs to completion may lead to errors in the Corpus.
    #     :param obj_type: 'utterance', 'conversation', 'speaker'
    #     :param attribute: name of metadata attribute
    #     :return: None
    #     """
    #     self.meta_index.lock_metadata_deletion[obj_type] = False
    #     for obj in self.iter_objs(obj_type):
    #         if attribute in obj.meta:
    #             del obj.meta[attribute]
    #     self.meta_index.del_from_index(obj_type, attribute)
    #     self.meta_index.lock_metadata_deletion[obj_type] = True

    # def set_vector_matrix(
    #     self, name: str, matrix, ids: List[str] = None, columns: List[str] = None
    # ):
    #     """
    #     Adds a vector matrix to the Corpus, where the matrix is an array of vector representations of some
    #     set of Corpus components (i.e. Utterances, Conversations, Speakers).
    #     A ConvoKitMatrix object is initialized from the arguments and stored in the Corpus.
    #     :param name: descriptive name for the matrix
    #     :param matrix: numpy or scipy array matrix
    #     :param ids: optional list of Corpus component object ids, where each id corresponds to each row of the matrix
    #     :param columns: optional list of names for the columns of the matrix
    #     :return: None
    #     """

    #     matrix = ConvoKitMatrix(name=name, matrix=matrix, ids=ids, columns=columns)
    #     if name in self.meta_index.vectors:
    #         warn(
    #             'Vector matrix "{}" already exists. Overwriting it with newly set vector matrix.'.format(
    #                 name
    #             )
    #         )
    #     self.meta_index.add_vector(name)
    #     self._vector_matrices[name] = matrix

    # def append_vector_matrix(self, matrix: ConvoKitMatrix):
    #     """
    #     Adds an already constructed ConvoKitMatrix to the Corpus.
    #     :param matrix: a ConvoKitMatrix object
    #     :return: None
    #     """
    #     if matrix.name in self.meta_index.vectors:
    #         warn(
    #             'Vector matrix "{}" already exists. '
    #             "Overwriting it with newly appended vector matrix that has name: '{}'.".format(
    #                 matrix.name, matrix.name
    #             )
    #         )
    #     self.meta_index.add_vector(matrix.name)
    #     self._vector_matrices[matrix.name] = matrix

    # def get_vector_matrix(self, name):
    #     """
    #     Gets the ConvoKitMatrix stored in the corpus as `name`.
    #     Returns None if no such matrix exists.
    #     :param name: name of the vector matrix
    #     :return: a ConvoKitMatrix object
    #     """
    #     # This is the lazy load step
    #     if name in self.vectors and name not in self._vector_matrices:
    #         matrix = ConvoKitMatrix.from_dir(self.corpus_dirpath, name)
    #         if matrix is not None:
    #             self._vector_matrices[name] = matrix
    #     return self._vector_matrices[name]

    # def get_vectors(
    #     self,
    #     name,
    #     ids: Optional[List[str]] = None,
    #     columns: Optional[List[str]] = None,
    #     as_dataframe: bool = False,
    # ):
    #     """
    #     Get the vectors for some corpus component objects.
    #     :param name: name of the vector matrix
    #     :param ids: optional list of object ids to get vectors for; all by default
    #     :param columns: optional list of named columns of the vector to include; all by default
    #     :param as_dataframe: whether to return the vector as a dataframe (True) or in its raw array form (False). False
    #         by default.
    #     :return: a vector matrix (either np.ndarray or csr_matrix) or a pandas dataframe
    #     """
    #     return self.get_vector_matrix(name).get_vectors(
    #         ids=ids, columns=columns, as_dataframe=as_dataframe
    #     )

    # def delete_vector_matrix(self, name):
    #     """
    #     Deletes the vector matrix stored under `name`.
    #     :param name: name of the vector mtrix
    #     :return: None
    #     """
    #     self.meta_index.vectors.remove(name)
    #     if name in self._vector_matrices:
    #         del self._vector_matrices[name]

    # def dump_vectors(self, name, dir_name=None):

    #     if (self.corpus_dirpath is None) and (dir_name is None):
    #         raise ValueError("Must specify a directory to read from.")
    #     if dir_name is None:
    #         dir_name = self.corpus_dirpath

    #     self.get_vector_matrix(name).dump(dir_name)

    # def load_info(self, obj_type, fields=None, dir_name=None):
    #     """
    #     loads attributes of objects in a corpus from disk.
    #     This function, along with dump_info, supports cases where a particular attribute is to be stored separately from
    #     the other corpus files, for organization or efficiency. These attributes will not be read when the corpus is
    #     initialized; rather, they can be loaded on-demand using this function.
    #     For each attribute with name <NAME>, will read from a file called info.<NAME>.jsonl, and load each attribute
    #     value into the respective object's .meta field.
    #     :param obj_type: type of object the attribute is associated with. can be one of "utterance", "speaker", "conversation".
    #     :param fields: a list of names of attributes to load. if empty, will load all attributes stored in the specified directory dir_name.
    #     :param dir_name: the directory to read attributes from. by default, or if set to None, will read from the directory that the Corpus was loaded from.
    #     :return: None
    #     """
    #     if fields is None:
    #         fields = []

    #     if (self.corpus_dirpath is None) and (dir_name is None):
    #         raise ValueError("Must specify a directory to read from.")
    #     if dir_name is None:
    #         dir_name = self.corpus_dirpath

    #     if len(fields) == 0:
    #         fields = [
    #             x.replace("info.", "").replace(".jsonl", "")
    #             for x in os.listdir(dir_name)
    #             if x.startswith("info")
    #         ]

    #     for field in fields:
    #         # self.aux_info[field] = self.load_jsonlist_to_dict(
    #         #     os.path.join(dir_name, 'feat.%s.jsonl' % field))
    #         if self.storage_type == "mem":
    #             load_info_to_mem(self, dir_name, obj_type, field)
    #         elif self.storage_type == "db":
    #             load_info_to_db(self, dir_name, obj_type, field)

    # def dump_info(self, obj_type, fields, dir_name=None):
    #     """
    #     writes attributes of objects in a corpus to disk.
    #     This function, along with load_info, supports cases where a particular attribute is to be stored separately from the other corpus files, for organization or efficiency. These attributes will not be read when the corpus is initialized; rather, they can be loaded on-demand using this function.
    #     For each attribute with name <NAME>, will write to a file called info.<NAME>.jsonl, where rows are json-serialized dictionaries structured as {"id": id of object, "value": value of attribute}.
    #     :param obj_type: type of object the attribute is associated with. can be one of "utterance", "speaker", "conversation".
    #     :param fields: a list of names of attributes to write to disk.
    #     :param dir_name: the directory to write attributes to. by default, or if set to None, will read from the directory that the Corpus was loaded from.
    #     :return: None
    #     """

    #     if (self.corpus_dirpath is None) and (dir_name is None):
    #         raise ValueError("must specify a directory to write to")

    #     if dir_name is None:
    #         dir_name = self.corpus_dirpath
    #     # if len(fields) == 0:
    #     #     fields = self.aux_info.keys()
    #     for field in fields:
    #         # if field not in self.aux_info:
    #         #     raise ValueError("field %s not in index" % field)
    #         iterator = self.iter_objs(obj_type)
    #         entries = {obj.id: obj.retrieve_meta(field) for obj in iterator}
    #         # self.dump_jsonlist_from_dict(self.aux_info[field],
    #         #     os.path.join(dir_name, 'feat.%s.jsonl' % field))
    #         dump_jsonlist_from_dict(entries, os.path.join(dir_name, "info.%s.jsonl" % field))

    # def get_attribute_table(self, obj_type, attrs):
    #     """
    #     returns a DataFrame, indexed by the IDs of objects of `obj_type`, containing attributes of these objects.
    #     :param obj_type: the type of object to get attributes for. can be `'utterance'`, `'speaker'` or `'conversation'`.
    #     :param attrs: a list of names of attributes to get.
    #     :return: a Pandas DataFrame of attributes.
    #     """
    #     iterator = self.iter_objs(obj_type)

    #     table_entries = []
    #     for obj in iterator:
    #         entry = dict()
    #         entry["id"] = obj.id
    #         for attr in attrs:
    #             entry[attr] = obj.retrieve_meta(attr)
    #         table_entries.append(entry)
    #     return pd.DataFrame(table_entries).set_index("id")

    # def set_speaker_convo_info(self, speaker_id, convo_id, key, value):
    #     """
    #     assigns speaker-conversation attribute `key` with `value` to speaker `speaker_id` in conversation `convo_id`.
    #     :param speaker_id: speaker
    #     :param convo_id: conversation
    #     :param key: name of attribute
    #     :param value: value of attribute
    #     :return: None
    #     """

    #     speaker = self.get_speaker(speaker_id)
    #     speaker_convos = speaker.meta.get("conversations", {})
    #     if convo_id not in speaker_convos:
    #         speaker_convos[convo_id] = {}
    #     speaker_convos[convo_id][key] = value
    #     speaker.meta["conversations"] = speaker_convos

    # def get_speaker_convo_info(self, speaker_id, convo_id, key=None):
    #     """
    #     retreives speaker-conversation attribute `key` for `speaker_id` in conversation `convo_id`.
    #     :param speaker_id: speaker
    #     :param convo_id: conversation
    #     :param key: name of attribute. if None, will return all attributes for that speaker-conversation.
    #     :return: attribute value
    #     """

    #     speaker = self.get_speaker(speaker_id)
    #     if "conversations" not in speaker.meta:
    #         return None
    #     if key is None:
    #         return speaker.meta["conversations"].get(convo_id, {})
    #     return speaker.meta["conversations"].get(convo_id, {}).get(key)

    # def organize_speaker_convo_history(self, utterance_filter=None):
    #     """
    #     For each speaker, pre-computes a list of all of their utterances, organized by the conversation they participated in. Annotates speaker with the following:
    #         * `n_convos`: number of conversations
    #         * `start_time`: time of first utterance, across all conversations
    #         * `conversations`: a dictionary keyed by conversation id, where entries consist of:
    #             * `idx`: the index of the conversation, in terms of the time of the first utterance contributed by that particular speaker (i.e., `idx=0` means this is the first conversation the speaker ever participated in)
    #             * `n_utterances`: the number of utterances the speaker contributed in the conversation
    #             * `start_time`: the timestamp of the speaker's first utterance in the conversation
    #             * `utterance_ids`: a list of ids of utterances contributed by the speaker, ordered by timestamp.
    #         In case timestamps are not provided with utterances, the present behavior is to sort just by utterance id.
    #     :param utterance_filter: function that returns True for an utterance that counts towards a speaker having participated in that conversation. (e.g., one could filter out conversations where the speaker contributed less than k words per utterance)
    #     """

    #     if utterance_filter is None:
    #         utterance_filter = lambda x: True
    #     else:
    #         utterance_filter = utterance_filter

    #     speaker_to_convo_utts = defaultdict(lambda: defaultdict(list))
    #     for utterance in self.iter_utterances():
    #         if not utterance_filter(utterance):
    #             continue

    #         speaker_to_convo_utts[utterance.speaker.id][utterance.conversation_id].append(
    #             (utterance.id, utterance.timestamp)
    #         )
    #     for speaker, convo_utts in speaker_to_convo_utts.items():
    #         for convo, utts in convo_utts.items():
    #             sorted_utts = sorted(utts, key=lambda x: (x[1], x[0]))
    #             self.set_speaker_convo_info(
    #                 speaker, convo, "utterance_ids", [x[0] for x in sorted_utts]
    #             )
    #             self.set_speaker_convo_info(speaker, convo, "start_time", sorted_utts[0][1])
    #             self.set_speaker_convo_info(speaker, convo, "n_utterances", len(sorted_utts))
    #     for speaker in self.iter_speakers():
    #         try:
    #             speaker.add_meta("n_convos", len(speaker.retrieve_meta("conversations")))
    #         except:
    #             continue

    #         sorted_convos = sorted(
    #             speaker.retrieve_meta("conversations").items(),
    #             key=lambda x: (x[1]["start_time"], x[1]["utterance_ids"][0]),
    #         )
    #         speaker.add_meta("start_time", sorted_convos[0][1]["start_time"])
    #         for idx, (convo_id, _) in enumerate(sorted_convos):
    #             self.set_speaker_convo_info(speaker.id, convo_id, "idx", idx)

    # def get_speaker_convo_attribute_table(self, attrs):
    #     """
    #     Returns a table where each row lists a (speaker, convo) level aggregate for each attribute in attrs.
    #     :param attrs: list of (speaker, convo) attribute names
    #     :return: DataFrame containing all speaker,convo attributes.
    #     """

    #     table_entries = []
    #     for speaker in self.iter_speakers():
    #         if "conversations" not in speaker.meta:
    #             continue
    #         for convo_id, convo_dict in speaker.meta["conversations"].items():
    #             entry = {
    #                 "id": "%s__%s" % (speaker.id, convo_id),
    #                 "speaker": speaker.id,
    #                 "convo_id": convo_id,
    #                 "convo_idx": convo_dict["idx"],
    #             }

    #             for attr in attrs:
    #                 entry[attr] = convo_dict.get(attr, None)
    #             table_entries.append(entry)
    #     return pd.DataFrame(table_entries).set_index("id")

    # def get_full_attribute_table(
    #     self,
    #     speaker_convo_attrs,
    #     speaker_attrs=None,
    #     convo_attrs=None,
    #     speaker_suffix="__speaker",
    #     convo_suffix="__convo",
    # ):
    #     """
    #     Returns a table where each row lists a (speaker, convo) level aggregate for each attribute in attrs,
    #     along with speaker-level and conversation-level attributes; by default these attributes are suffixed with
    #     '__speaker' and '__convo' respectively.
    #     :param speaker_convo_attrs: list of (speaker, convo) attribute names
    #     :param speaker_attrs: list of speaker attribute names
    #     :param convo_attrs: list of conversation attribute names
    #     :param speaker_suffix: suffix to append to names of speaker-level attributes
    #     :param convo_suffix: suffix to append to names of conversation-level attributes.
    #     :return: DataFrame containing all attributes.
    #     """
    #     if speaker_attrs is None:
    #         speaker_attrs = []
    #     if convo_attrs is None:
    #         convo_attrs = []

    #     uc_df = self.get_speaker_convo_attribute_table(speaker_convo_attrs)
    #     u_df = self.get_attribute_table("speaker", speaker_attrs)
    #     u_df.columns = [x + speaker_suffix for x in u_df.columns]
    #     c_df = self.get_attribute_table("conversation", convo_attrs)
    #     c_df.columns = [x + convo_suffix for x in c_df.columns]
    #     return uc_df.join(u_df, on="speaker").join(c_df, on="convo_id")

    # def update_metadata_from_df(self, obj_type, df):
    #     assert obj_type in ["utterance", "speaker", "conversation"]
    #     meta_cols = extract_meta_from_df(df)
    #     df.columns = [col.replace("meta.", "") for col in df.columns]
    #     df = df.set_index("id")
    #     for obj in self.iter_objs(obj_type):
    #         obj_meta = df.loc[obj.id][meta_cols].to_dict() if meta_cols else None
    #         if obj_meta is not None:
    #             obj.meta.update(obj_meta)
    #     return self


#### convokit.CorpusHelper

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


def check_id_for_mongodb(corpus_id):
    # List of collection name restrictions from official MongoDB docs:
    # https://www.mongodb.com/docs/manual/reference/limits/#mongodb-limit-Restriction-on-Collection-Names
    if "$" in corpus_id:
        return "contains the restricted character '$'"
    if len(corpus_id) == 0:
        return "string is empty"
    if "\0" in corpus_id:
        return "contains a null character"
    if "system." in corpus_id:
        return 'starts with the restricted prefix "system."'
    if not (corpus_id[0] == "_" or corpus_id[0].isalpha()):
        return "name must start with an underscore or letter character"
    return None


def get_corpus_dirpath(filename: str) -> Optional[str]:
    if filename is None:
        return None
    elif os.path.isdir(filename):
        return filename
    else:
        return os.path.dirname(filename)


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


def load_utterance_info_from_dir(
    dirname, utterance_start_index, utterance_end_index, exclude_utterance_meta
):
    assert dirname is not None
    assert os.path.isdir(dirname)

    if utterance_start_index is None:
        utterance_start_index = 0
    if utterance_end_index is None:
        utterance_end_index = float("inf")

    if os.path.exists(os.path.join(dirname, "utterances.jsonl")):
        with open(os.path.join(dirname, "utterances.jsonl"), "r") as f:
            utterances = []
            idx = 0
            for line in f:
                if utterance_start_index <= idx <= utterance_end_index:
                    utterances.append(json.loads(line))
                idx += 1

    elif os.path.exists(os.path.join(dirname, "utterances.json")):
        with open(os.path.join(dirname, "utterances.json"), "r") as f:
            utterances = json.load(f)

    if exclude_utterance_meta:
        for utt in utterances:
            for field in exclude_utterance_meta:
                del utt["meta"][field]

    return utterances


def load_speakers_data_from_dir(filename, exclude_speaker_meta):
    speaker_file = "speakers.json" if "speakers.json" in os.listdir(filename) else "users.json"
    with open(os.path.join(filename, speaker_file), "r") as f:
        id_to_speaker_data = json.load(f)

        if (
            len(id_to_speaker_data) > 0
            and len(next(iter(id_to_speaker_data.values())))
            and "vectors" in id_to_speaker_data == 2
        ):
            # has vectors data
            for _, speaker_data in id_to_speaker_data.items():
                for k in exclude_speaker_meta:
                    if k in speaker_data["meta"]:
                        del speaker_data["meta"][k]
        else:
            for _, speaker_data in id_to_speaker_data.items():
                for k in exclude_speaker_meta:
                    if k in speaker_data:
                        del speaker_data[k]
    return id_to_speaker_data


def load_convos_data_from_dir(filename, exclude_conversation_meta):
    """
    :param filename:
    :param exclude_conversation_meta:
    :return: a mapping from convo id to convo meta
    """
    with open(os.path.join(filename, "conversations.json"), "r") as f:
        id_to_convo_data = json.load(f)

        if (
            len(id_to_convo_data) > 0
            and len(next(iter(id_to_convo_data.values())))
            and "vectors" in id_to_convo_data == 2
        ):
            # has vectors data
            for _, convo_data in id_to_convo_data.items():
                for k in exclude_conversation_meta:
                    if k in convo_data["meta"]:
                        del convo_data["meta"][k]
        else:
            for _, convo_data in id_to_convo_data.items():
                for k in exclude_conversation_meta:
                    if k in convo_data:
                        del convo_data[k]
    return id_to_convo_data


def load_corpus_meta_from_dir(filename, corpus_meta, exclude_overall_meta):
    """
    Updates corpus meta object with fields from corpus.json
    """
    with open(os.path.join(filename, "corpus.json"), "r") as f:
        for k, v in json.load(f).items():
            if k in exclude_overall_meta:
                continue
            corpus_meta[k] = v


def unpack_binary_data_for_utts(utterances, filename, utterance_index, exclude_meta, KeyMeta):
    """
    :param utterances: mapping from utterance id to {'meta': ..., 'vectors': ...}
    :param filename: filepath containing corpus files
    :param utterance_index: utterance meta index
    :param exclude_meta: list of metadata attributes to exclude
    :param KeyMeta: name of metadata key, should be 'meta'
    :return:
    """
    for field, field_types in utterance_index.items():
        if len(field_types) > 0 and field_types[0] == "bin" and field not in exclude_meta:
            with open(os.path.join(filename, field + "-bin.p"), "rb") as f:
                l_bin = pickle.load(f)
            for i, ut in enumerate(utterances):
                for k, v in ut[KeyMeta].items():
                    if (
                        k == field
                        and type(v) == str
                        and v.startswith(BIN_DELIM_L)
                        and v.endswith(BIN_DELIM_R)
                    ):
                        idx = int(v[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                        utterances[i][KeyMeta][k] = l_bin[idx]
    for field in exclude_meta:
        del utterance_index[field]


def unpack_binary_data(filename, objs_data, object_index, obj_type, exclude_meta):
    """
    Unpack binary data for Speakers or Conversations
    :param filename: filepath containing the corpus data
    :param objs_data: a mapping from object id to a dictionary with two keys: 'meta' and 'vectors';
        in older versions, this is a mapping from object id to the metadata dictionary
    :param object_index: the meta_index dictionary for the component type
    :param obj_type: object type (i.e. speaker or conversation)
    :param exclude_meta: list of metadata attributes to exclude
    :return: None (mutates objs_data)
    """
    """
    Unpack binary data for Speakers or Conversations
    """
    # unpack speaker meta
    for field, field_types in object_index.items():
        if len(field_types) > 0 and field_types[0] == "bin" and field not in exclude_meta:
            with open(os.path.join(filename, field + "-{}-bin.p".format(obj_type)), "rb") as f:
                l_bin = pickle.load(f)
            for obj, data in objs_data.items():
                metadata = data["meta"] if len(data) == 2 and "vectors" in data else data
                for k, v in metadata.items():
                    if (
                        k == field
                        and type(v) == str
                        and str(v).startswith(BIN_DELIM_L)
                        and str(v).endswith(BIN_DELIM_R)
                    ):
                        idx = int(v[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                        metadata[k] = l_bin[idx]
    for field in exclude_meta:
        del object_index[field]


def unpack_all_binary_data(
    filename: str,
    meta_index: ConvoKitIndex,
    meta: ConvoKitMeta,
    utterances: List[Utterance],
    speakers_data: Dict[str, Dict],
    convos_data: Dict[str, Dict],
    exclude_utterance_meta: List[str],
    exclude_speaker_meta: List[str],
    exclude_conversation_meta: List[str],
    exclude_overall_meta: List[str],
):
    # unpack binary data for utterances
    unpack_binary_data_for_utts(
        utterances,
        filename,
        meta_index.utterances_index,
        exclude_utterance_meta,
        KeyMeta,
    )
    # unpack binary data for speakers
    unpack_binary_data(
        filename,
        speakers_data,
        meta_index.speakers_index,
        "speaker",
        exclude_speaker_meta,
    )

    # unpack binary data for conversations
    unpack_binary_data(
        filename,
        convos_data,
        meta_index.conversations_index,
        "convo",
        exclude_conversation_meta,
    )

    # unpack binary data for overall corpus
    unpack_binary_data(
        filename,
        meta,
        meta_index.overall_index,
        "overall",
        exclude_overall_meta,
    )


def load_from_utterance_file(filename, utterance_start_index, utterance_end_index):
    """
    where filename is "utterances.json" or "utterances.jsonl" for example
    """
    with open(filename, "r") as f:
        try:
            ext = filename.split(".")[-1]
            if ext == "json":
                utterances = json.load(f)
            elif ext == "jsonl":
                utterances = []
                if utterance_start_index is None:
                    utterance_start_index = 0
                if utterance_end_index is None:
                    utterance_end_index = float("inf")
                idx = 0
                for line in f:
                    if utterance_start_index <= idx <= utterance_end_index:
                        utterances.append(json.loads(line))
                    idx += 1
        except Exception as e:
            raise Exception(
                "Could not load corpus. Expected json file, encountered error: \n" + str(e)
            )
    return utterances


def initialize_speakers_and_utterances_objects(corpus, utterances, speakers_data):
    """
    Initialize Speaker and Utterance objects
    """
    if len(utterances) > 0:  # utterances might be empty for invalid corpus start/end indices
        KeySpeaker = "speaker" if "speaker" in utterances[0] else "user"
        KeyConvoId = "conversation_id" if "conversation_id" in utterances[0] else "root"

    for i, u in enumerate(utterances):
        u = defaultdict(lambda: None, u)
        speaker_key = u[KeySpeaker]
        if speaker_key not in corpus.speakers:
            if u[KeySpeaker] not in speakers_data:
                warn(
                    "CorpusLoadWarning: Missing speaker metadata for speaker ID: {}. "
                    "Initializing default empty metadata instead.".format(u[KeySpeaker])
                )
                speakers_data[u[KeySpeaker]] = {}
            if KeyMeta in speakers_data[u[KeySpeaker]]:
                corpus.speakers[speaker_key] = Speaker(
                    owner=corpus, id=u[KeySpeaker], meta=speakers_data[u[KeySpeaker]][KeyMeta]
                )
            else:
                corpus.speakers[speaker_key] = Speaker(
                    owner=corpus, id=u[KeySpeaker], meta=speakers_data[u[KeySpeaker]]
                )

        speaker = corpus.speakers[speaker_key]
        speaker.vectors = speakers_data[u[KeySpeaker]].get(KeyVectors, [])

        # temp fix for reddit reply_to
        if "reply_to" in u:
            reply_to_data = u["reply_to"]
        else:
            reply_to_data = u[KeyReplyTo]
        utt = Utterance(
            owner=corpus,
            id=u[KeyId],
            speaker=speaker,
            conversation_id=u[KeyConvoId],
            reply_to=reply_to_data,
            timestamp=u[KeyTimestamp],
            text=u[KeyText],
            meta=u[KeyMeta],
        )
        utt.vectors = u.get(KeyVectors, [])
        corpus.utterances[utt.id] = utt


def merge_utterance_lines(utt_dict):
    """
    For merging adjacent utterances by the same speaker
    """
    new_utterances = {}
    merged_with = {}
    for uid, utt in utt_dict.items():
        merged = False
        if utt.reply_to is not None and utt.speaker is not None:
            u0 = utt_dict[utt.reply_to]
            if u0.conversation_id == utt.conversation_id and u0.speaker == utt.speaker:
                merge_target = merged_with[u0.id] if u0.id in merged_with else u0.id
                new_utterances[merge_target].text += " " + utt.text
                merged_with[utt.id] = merge_target
                merged = True
        if not merged:
            if utt.reply_to in merged_with:
                utt.reply_to = merged_with[utt.reply_to]
            new_utterances[utt.id] = utt
    return new_utterances


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


def load_jsonlist_to_dict(filename, index_key="id", value_key="value"):
    entries = {}
    with open(filename, "r") as f:
        for line in f:
            entry = json.loads(line)
            entries[entry[index_key]] = entry[value_key]
    return entries


def dump_jsonlist_from_dict(entries, filename, index_key="id", value_key="value"):
    with open(filename, "w") as f:
        for k, v in entries.items():
            json.dump({index_key: k, value_key: v}, f)
            f.write("\n")


def extract_meta_from_df(df):
    meta_cols = [col.split(".")[1] for col in df if col.startswith("meta")]
    return meta_cols


def load_binary_metadata(filename, index, exclude_meta=None):
    binary_data = {"utterance": {}, "conversation": {}, "speaker": {}, "corpus": {}}
    for component_type in binary_data:
        meta_index = index.get_index(component_type)
        for meta_key, meta_type in meta_index.items():
            if meta_type == ["bin"] and (
                exclude_meta is None or meta_key not in exclude_meta[component_type]
            ):
                # filename format differs for utterances versus everything else
                filename_suffix = (
                    "-bin.p"
                    if component_type == "utterance"
                    else "-{}-bin.p".format(component_type)
                )
                try:
                    with open(os.path.join(filename, meta_key + filename_suffix), "rb") as f:
                        l_bin = pickle.load(f)
                        binary_data[component_type][meta_key] = l_bin
                except FileNotFoundError:
                    warn(
                        f"Metadata field {meta_key} is specified to have binary type but no saved binary data was found. This field will be skipped."
                    )
                    # update the exclude_meta list to force this field to get skipped
                    # in the subsequent corpus loading logic
                    if exclude_meta is None:
                        exclude_meta = defaultdict(list)
                    exclude_meta[component_type].append(meta_key)
    return binary_data, exclude_meta


def load_jsonlist_to_db(
    filename,
    db,
    collection_prefix,
    start_line=None,
    end_line=None,
    exclude_meta=None,
    bin_meta=None,
):
    """
    Populate the specified MongoDB database with the utterance data contained in
    the given filename (which should point to an utterances.jsonl file).
    """
    utt_collection = db[f"{collection_prefix}_utterance"]
    meta_collection = db[f"{collection_prefix}_meta"]
    inserted_ids = set()
    speaker_key = None
    convo_key = None
    reply_key = None
    with open(filename) as f:
        utt_insertion_buffer = []
        meta_insertion_buffer = []
        for ln, line in enumerate(f):
            if start_line is not None and ln < start_line:
                continue
            if end_line is not None and ln > end_line:
                break
            utt_obj = json.loads(line)
            if speaker_key is None:
                # backwards compatibility for corpora made before the user->speaker rename
                speaker_key = "speaker" if "speaker" in utt_obj else "user"
            if convo_key is None:
                # backwards compatibility for corpora made before the root->conversation_id rename
                convo_key = "conversation_id" if "conversation_id" in utt_obj else "root"
            if reply_key is None:
                # fix for misnamed reply_to in subreddit corpora
                reply_key = "reply-to" if "reply-to" in utt_obj else "reply_to"
            utt_obj = defaultdict(lambda: None, utt_obj)
            utt_insertion_buffer.append(
                UpdateOne(
                    {"_id": utt_obj["id"]},
                    {
                        "$set": {
                            "speaker_id": utt_obj[speaker_key],
                            "conversation_id": utt_obj[convo_key],
                            "reply_to": utt_obj[reply_key],
                            "timestamp": utt_obj["timestamp"],
                            "text": utt_obj["text"],
                        }
                    },
                    upsert=True,
                )
            )
            utt_meta = utt_obj["meta"]
            if exclude_meta is not None:
                for exclude_key in exclude_meta:
                    if exclude_key in utt_meta:
                        del utt_meta[exclude_key]
            if bin_meta is not None:
                for key, bin_list in bin_meta.items():
                    bin_locator = utt_meta.get(key, None)
                    if (
                        type(bin_locator) == str
                        and bin_locator.startswith(BIN_DELIM_L)
                        and bin_locator.endswith(BIN_DELIM_R)
                    ):
                        bin_idx = int(bin_locator[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                        utt_meta[key] = bson.Binary(pickle.dumps(bin_list[bin_idx]))
            meta_insertion_buffer.append(
                UpdateOne({"_id": "utterance_" + utt_obj["id"]}, {"$set": utt_meta}, upsert=True)
            )
            inserted_ids.add(utt_obj["id"])
            if len(utt_insertion_buffer) >= JSONLIST_BUFFER_SIZE:
                utt_collection.bulk_write(utt_insertion_buffer)
                meta_collection.bulk_write(meta_insertion_buffer)
                utt_insertion_buffer = []
                meta_insertion_buffer = []
        # after loop termination, insert any remaining items in the buffer
        if len(utt_insertion_buffer) > 0:
            utt_collection.bulk_write(utt_insertion_buffer)
            meta_collection.bulk_write(meta_insertion_buffer)
            utt_insertion_buffer = []
            meta_insertion_buffer = []
    return inserted_ids


def load_json_to_db(
    filename, db, collection_prefix, component_type, exclude_meta=None, bin_meta=None
):
    """
    Populate the specified MongoDB database with corpus component data from
    either the speakers.json or conversations.json file located in a directory
    containing valid ConvoKit Corpus data. The component_type parameter controls
    which JSON file gets used.
    """
    component_collection = db[f"{collection_prefix}_{component_type}"]
    meta_collection = db[f"{collection_prefix}_meta"]
    if component_type == "speaker":
        json_data = load_speakers_data_from_dir(filename, exclude_meta)
    elif component_type == "conversation":
        json_data = load_convos_data_from_dir(filename, exclude_meta)
    component_insertion_buffer = []
    meta_insertion_buffer = []
    for component_id, component_data in json_data.items():
        if KeyMeta in component_data:
            # contains non-metadata entries
            payload = {k: v for k, v in component_data.items() if k not in {"meta", "vectors"}}
            meta = component_data[KeyMeta]
        else:
            # contains only metadata, with metadata at the top level
            payload = {}
            meta = component_data
        component_insertion_buffer.append(
            UpdateOne({"_id": component_id}, {"$set": payload}, upsert=True)
        )
        if bin_meta is not None:
            for key, bin_list in bin_meta.items():
                bin_locator = meta.get(key, None)
                if (
                    type(bin_locator) == str
                    and bin_locator.startswith(BIN_DELIM_L)
                    and bin_locator.endswith(BIN_DELIM_R)
                ):
                    bin_idx = int(bin_locator[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                    meta[key] = bson.Binary(pickle.dumps(bin_list[bin_idx]))
        meta_insertion_buffer.append(
            UpdateOne({"_id": f"{component_type}_{component_id}"}, {"$set": meta}, upsert=True)
        )
    component_collection.bulk_write(component_insertion_buffer)
    meta_collection.bulk_write(meta_insertion_buffer)


def load_corpus_info_to_db(filename, db, collection_prefix, exclude_meta=None, bin_meta=None):
    """
    Populate the specified MongoDB database with Corpus metadata loaded from the
    corpus.json file of a directory containing valid ConvoKit Corpus data.
    """
    if exclude_meta is None:
        exclude_meta = {}
    meta_collection = db[f"{collection_prefix}_meta"]
    with open(os.path.join(filename, "corpus.json")) as f:
        corpus_meta = {k: v for k, v in json.load(f).items() if k not in exclude_meta}
        if bin_meta is not None:
            for key, bin_list in bin_meta.items():
                bin_locator = corpus_meta.get(key, None)
                if (
                    type(bin_locator) == str
                    and bin_locator.startswith(BIN_DELIM_L)
                    and bin_locator.endswith(BIN_DELIM_R)
                ):
                    bin_idx = int(bin_locator[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                    corpus_meta[key] = bson.Binary(pickle.dumps(bin_list[bin_idx]))
        meta_collection.update_one(
            {"_id": f"corpus_{collection_prefix}"}, {"$set": corpus_meta}, upsert=True
        )


def load_info_to_mem(corpus, dir_name, obj_type, field):
    """
    Helper for load_info in mem mode that reads the file for the specified extra
    info field, loads it into memory, and assigns the entries to their
    corresponding corpus components.
    """
    getter = lambda oid: corpus.get_object(obj_type, oid)
    entries = load_jsonlist_to_dict(os.path.join(dir_name, "info.%s.jsonl" % field))
    for k, v in entries.items():
        try:
            obj = getter(k)
            obj.add_meta(field, v)
        except:
            continue


def load_info_to_db(corpus, dir_name, obj_type, field, index_key="id", value_key="value"):
    """
    Helper for load_info in DB mode that reads the jsonlist file for the
    specified extra info field in a batched line-by-line manner, populates
    its contents into the DB, and updates the Corpus' metadata index
    """
    filename = os.path.join(dir_name, "info.%s.jsonl" % field)
    meta_collection = corpus.storage.get_collection("meta")

    # attept to use saved type information
    index_file = os.path.join(dir_name, "index.json")
    with open(index_file) as f:
        raw_index = json.load(f)
        try:
            field_type = raw_index[f"{obj_type}s-index"][field]
            corpus.meta_index.get_index(obj_type)[field] = field_type
            index_updated = True
        except:
            # field not recorded in the index file; we will need to infer
            # types during insertion time
            index_updated = False

    # iteratively insert the info in the DB in batched fashion
    with open(filename) as f:
        info_insertion_buffer = []
        for line in f:
            info_json = json.loads(line)
            obj_id, info_val = info_json[index_key], info_json[value_key]
            if not index_updated:
                # we were previously unable to fetch the type info from the
                # index file, so we must infer it now
                ConvoKitMeta._check_type_and_update_index(
                    corpus.meta_index, obj_type, field, info_val
                )
            info_insertion_buffer.append(
                UpdateOne(
                    {"_id": "{}_{}".format(obj_type, obj_id)},
                    {"$set": {field: info_val}},
                    upsert=True,
                )
            )
            if len(info_insertion_buffer) >= JSONLIST_BUFFER_SIZE:
                meta_collection.bulk_write(info_insertion_buffer)
                info_insertion_buffer = []
        # after loop termination, insert any remaining items in the buffer
        if len(info_insertion_buffer) > 0:
            meta_collection.bulk_write(info_insertion_buffer)
            info_insertion_buffer = []


def clean_up_excluded_meta(meta_index, exclude_meta):
    """
    Remove excluded metadata from the metadata index
    """
    for component_type, excluded_keys in exclude_meta.items():
        for key in excluded_keys:
            meta_index.del_from_index(component_type, key)


def populate_db_from_file(
    filename,
    db,
    collection_prefix,
    meta_index,
    utterance_start_index,
    utterance_end_index,
    exclude_utterance_meta,
    exclude_conversation_meta,
    exclude_speaker_meta,
    exclude_overall_meta,
):
    """
    Populate all necessary collections of a MongoDB database so that it can be
    used by a DBStorageManager, sourcing data from the valid ConvoKit Corpus
    data pointed to by the filename parameter.
    """
    binary_meta, updated_exclude_meta = load_binary_metadata(
        filename,
        meta_index,
        {
            "utterance": exclude_utterance_meta,
            "conversation": exclude_conversation_meta,
            "speaker": exclude_speaker_meta,
            "corpus": exclude_overall_meta,
        },
    )

    # exclusion lists may have changed if errors were encountered while loading
    # the binary metadata
    if updated_exclude_meta is not None:
        exclude_utterance_meta = updated_exclude_meta["utterance"]
        exclude_conversation_meta = updated_exclude_meta["conversation"]
        exclude_speaker_meta = updated_exclude_meta["speaker"]
        exclude_overall_meta = updated_exclude_meta["corpus"]

    # first load the utterance data
    inserted_utt_ids = load_jsonlist_to_db(
        os.path.join(filename, "utterances.jsonl"),
        db,
        collection_prefix,
        utterance_start_index,
        utterance_end_index,
        exclude_utterance_meta,
        binary_meta["utterance"],
    )
    # next load the speaker and conversation data
    for component_type in ["speaker", "conversation"]:
        load_json_to_db(
            filename,
            db,
            collection_prefix,
            component_type,
            (exclude_speaker_meta if component_type == "speaker" else exclude_conversation_meta),
            binary_meta[component_type],
        )
    # finally, load the corpus metadata
    load_corpus_info_to_db(
        filename, db, collection_prefix, exclude_overall_meta, binary_meta["corpus"]
    )

    # make sure skipped metadata isn't kept in the final index
    clean_up_excluded_meta(
        meta_index,
        {
            "utterance": exclude_utterance_meta,
            "conversation": exclude_conversation_meta,
            "speaker": exclude_speaker_meta,
            "corpus": exclude_overall_meta,
        },
    )

    return inserted_utt_ids


def init_corpus_from_storage_manager(corpus, utt_ids=None):
    """
    Use an already-populated MongoDB database to initialize the components of
    the specified Corpus (which should be empty before this function is called)
    """
    # we will bypass the initialization step when constructing components since
    # we know their necessary data already exists within the db
    corpus.storage.bypass_init = True

    # fetch object ids from the DB and initialize corpus components for them
    # create speakers first so we can refer to them when initializing utterances
    speakers = {}
    for speaker_doc in corpus.storage.data["speaker"].find(projection=["_id"]):
        speaker_id = speaker_doc["_id"]
        speakers[speaker_id] = Speaker(owner=corpus, id=speaker_id)
    corpus.speakers = speakers

    # next, create utterances
    utterances = {}
    convo_to_utts = defaultdict(list)
    for utt_doc in corpus.storage.data["utterance"].find(
        projection=["_id", "speaker_id", "conversation_id"]
    ):
        utt_id = utt_doc["_id"]
        if utt_ids is None or utt_id in utt_ids:
            convo_to_utts[utt_doc["conversation_id"]].append(utt_id)
            utterances[utt_id] = Utterance(
                owner=corpus, id=utt_id, speaker=speakers[utt_doc["speaker_id"]]
            )
    corpus.utterances = utterances

    # run post-construction integrity steps as in regular constructor
    corpus.conversations = initialize_conversations(corpus, {}, convo_to_utts)
    corpus.meta_index.enable_type_check()
    corpus.update_speakers_data()

    # restore the StorageManager's init behavior to default
    corpus.storage.bypass_init = False



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

        for (df_type, df) in [
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
    
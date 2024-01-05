"""Documentation about scikit-talk"""
import logging
# Import the modules that are part of the sktalk package
from .corpus.conversation import Conversation  # noqa: F401
from .corpus.corpus import Corpus  # noqa: F401
from .corpus.utterance import Utterance  # noqa: F401


logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Barbara Vreede"
__email__ = "b.vreede@esciencecenter.nl"
__version__ = "0.1.1"

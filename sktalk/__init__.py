"""Documentation about scikit-talk"""
import logging

# Import the modules that are part of the sktalk package
from .corpus.parsing.cha import ChaFile
from .corpus.corpus import Corpus
from .corpus.conversation import Conversation
from .corpus.utterance import Utterance

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Barbara Vreede"
__email__ = "b.vreede@esciencecenter.nl"
__version__ = "0.1.0"

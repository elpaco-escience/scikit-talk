import pandas as pd


class Corpus:
    """The Corpus class represents a conversation corpus.

    Args:
        path (str): The path to the CSV file containing the corpus data.

    Attributes:
        df (pandas.DataFrame): The pandas DataFrame containing corpus data.
        json (str): The JSON representation of the corpus data.
        speakers (list): A list of unique speakers extracted from the corpus.
    """

    def __init__(self, path):
        """Initialize a new instance of the Corpus class.

        Args:
            path (str): The path to the CSV file containing the corpus data.
        """
        self.df = pd.read_csv(path)
        self.json = self.df.to_json()

    def extract_speakers(self, speakercol='participant'):
        """Extract the unique speakers from the corpus data.

        Args:
            speakercol (str, optional): The name of the column containing
            the speaker information. Default is 'participant'.
        """
        speakers = self.df[speakercol]
        self.speakers = list(set(speakers))

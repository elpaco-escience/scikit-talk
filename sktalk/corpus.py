"""Documentation about the scikit-talk module - corpus."""
import pandas as pd


class Corpus:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.json = self.df.to_json()

    def extract_speakers(self):
        self.speakers = self.df['participant']


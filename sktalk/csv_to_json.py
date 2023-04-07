"""Documentation about the scikit-talk module - csv to json."""
import pandas as pd


class Corpus:
    def __init__(self, path):
        self.path = path

    def return_dataframe(self):
        self.df = pd.read_csv(self.path)

    def return_json(self):
        self.return_dataframe()
        self.json = self.df.to_json()

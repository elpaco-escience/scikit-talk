import abc
import json
from pathlib import Path
import numpy as np
import pandas as pd


class Writer(abc.ABC):
    @abc.abstractmethod
    def asdict(self):
        return NotImplemented

    def write_json(self, path: str = "./file.json"):
        """
        Write an object to a JSON file.

        Args:
            path (str): The path to the output file.
        """
        _path = Path(path).with_suffix(".json")

        object_dict = self.asdict()

        with open(_path, "w", encoding='utf-8') as file:
            json.dump(object_dict, file, indent=4)

    def write_csv(self, path: str = "./file.csv"):
        """Write the object to CSV files.

        Multiple csv files are created: one for the metadata (corpus and conversation level) and one for the utterances.
        The filenames are based on the provided path, with the following suffixes:
        - file_metadata.csv
        - file_utterances.csv

        Args:
            path (str, optional): Base name of csv output files. Defaults to "./file.csv".
        """
        _path = Path(path).with_suffix(".csv")

        self.metadata_df.to_csv(self._specify_path(
            _path, "metadata"), index=False)
        self.utterance_df.to_csv(self._specify_path(_path, "utterances"))

    def _specify_path(self, path: Path, specifier: str):
        return path.with_name(f"{path.stem}_{specifier}{path.suffix}")

    @classmethod
    def _metadata_to_df(cls, metadata: dict):
        norm = pd.json_normalize(data=metadata, sep="_")
        df = pd.DataFrame(norm)
        df[:] = np.vectorize(lambda x: ', '.join(
            x) if isinstance(x, list) else x)(df)
        return df

    @property
    def metadata_df(self):
        return NotImplemented

    @property
    def utterance_df(self):
        return NotImplemented

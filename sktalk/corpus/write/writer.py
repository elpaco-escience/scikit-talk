import abc
import csv
import json
from pathlib import Path
import pandas as pd
import numpy as np


class Writer(abc.ABC):
    @abc.abstractmethod
    def asdict(self):
        return NotImplemented

    def write_json(self, path: str = "./file.json"):
        """
        Write an object to a JSON file.

        Args:
            name (str): The file name
            directory (str): The path to the directory where the .json
            file will be saved.
        """
        _path = Path(path).with_suffix(".json")

        object_dict = self.asdict()

        with open(_path, "w", encoding='utf-8') as file:
            json.dump(object_dict, file, indent=4)

    def write_csv(self):
        return NotImplemented

    def _write_csv(self, path: str, headers: list, rows: list):
        _path = Path(path).with_suffix(".csv")

        with open(_path, "w", encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _specify_path(self, path: Path, specifier: str):
        return path.with_name(f"{path.stem}_{specifier}{path.suffix}")

    @classmethod
    def _metadata_to_df(cls, metadata: dict):
        norm = pd.json_normalize(data=metadata, sep="_")
        df = pd.DataFrame(norm)
        df[:] = np.vectorize(lambda x: ', '.join(x) if isinstance(x, list) else x)(df)
        return df

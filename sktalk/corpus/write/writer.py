import abc
import json
from pathlib import Path
import csv


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

    def write_csv(self, path: str = "./file.csv"):
        return NotImplemented
        

    def _write_csv(self, path: str, headers: list, rows: list):
        _path = Path(path).with_suffix(".csv")

        with open(_path, "w", encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                    writer.writerow(row)



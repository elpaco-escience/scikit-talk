import abc
import json
import os

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
        ext = os.path.splitext(path)[1]
        if ext != ".json":
            path += ".json"

        object_dict = self.asdict()

        with open(path, "w", encoding='utf-8') as file:
            json.dump(object_dict, file, indent=4)

import abc


class Writer(abc.ABC):
    @abc.abstractmethod
    def write(self):
        return NotImplemented

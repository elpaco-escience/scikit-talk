from .writer import Writer


class JsonWriter(Writer):
    def write(self):
        raise NotImplementedError()

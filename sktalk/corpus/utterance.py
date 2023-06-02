class Utterance:
    def __init__(self, conversation, metadata: dict) -> None:
        self._conversation = conversation
        self._metadata: metadata

    def get_audio(self):
        pass

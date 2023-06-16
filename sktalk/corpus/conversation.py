class Conversation:
    def __init__(
        self, utterances: list["Utterance"], metadata: dict = None  # noqa: F821
    ) -> None:
        self._utterances = utterances
        self._metadata = metadata

    @property
    def utterances(self):
        return self._utterances

    @property
    def metadata(self):
        return self._metadata

    def get_utterance(self, index) -> "Utterance":  # noqa: F821
        pass

    def summarize(self):
        for utterance in self._utterances:
            print(utterance)


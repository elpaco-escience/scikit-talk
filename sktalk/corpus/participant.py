from dataclasses import dataclass


@dataclass
class Participant:
    name: str  # noqa: E701
    age: int = None  # noqa: E701

from dataclasses import dataclass


@dataclass
class Participant:
    name: str
    age: int = None

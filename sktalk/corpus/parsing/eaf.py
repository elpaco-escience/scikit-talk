import re
import xml.etree.ElementTree as eTree
from ..utterance import Utterance
from .parser import InputFile


class EafFile(InputFile):
    """Parser for .eaf files."""
    # known values for the attributes
    TIER = "TIER"
    TIER_ID = "TIER_ID"
    ALIGNABLE_ANNOTATION = "ALIGNABLE_ANNOTATION"
    ANNOTATION_VALUE = "ANNOTATION_VALUE"
    TIME_SLOT_REF1 = "TIME_SLOT_REF1"
    TIME_SLOT_REF2 = "TIME_SLOT_REF2"
    TIME_ORDER = "TIME_ORDER"
    TIME_SLOT = "TIME_SLOT"
    TIME_SLOT_ID = "TIME_SLOT_ID"
    TIME_VALUE = "TIME_VALUE"

    def _extract_metadata(self):
        return {}

    def _extract_utterances(self):
        timedict = self._extract_times(self.root)

        # once again, a terrible loop >.<
        annotations, sorting = [], []

        for tier in self.root.findall(f".//{self.TIER}"):
            participant = tier.get(self.TIER_ID)

            for annotation in tier.findall(f".//{self.ALIGNABLE_ANNOTATION}"):
                utterance = annotation.find(self.ANNOTATION_VALUE).text

                begin = annotation.get(self.TIME_SLOT_REF1)
                end = annotation.get(self.TIME_SLOT_REF2)
                time = [timedict[begin], timedict[end]]

                annotations.append(
                    Utterance(
                        participant=participant,
                        utterance=utterance,
                        time=time
                    ))

                order = re.search(r"\d+", begin)
                sorting.append(int(order.group()))

        return [utt for _, utt in sorted(zip(sorting, annotations))]

    @property
    def root(self):
        if not hasattr(self, "_root"):
            tree = eTree.parse(self._path)
            self._root = tree.getroot()
        return self._root

    @staticmethod
    def _extract_times(root):
        time_slots = root.find(
            f".//{EafFile.TIME_ORDER}").findall(f".//{EafFile.TIME_SLOT}")
        return {stamp.get(EafFile.TIME_SLOT_ID): int(stamp.get(EafFile.TIME_VALUE)) for stamp in time_slots}

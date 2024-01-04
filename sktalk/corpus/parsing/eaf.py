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
        annotations = []
        for tier in self.root.findall(f".//{self.TIER}"):
            tier_id = tier.get(self.TIER_ID)
            utterances = [self._annotation_to_utterance(
                annotation, tier_id) for annotation in tier.findall(f".//{self.ALIGNABLE_ANNOTATION}")]
            annotations.extend(utterances)
        sorting = [utt.time[0] for utt in annotations]
        return [utt for _, utt in sorted(zip(sorting, annotations))]

    @property
    def root(self):
        if not hasattr(self, "_root"):
            tree = eTree.parse(self._path)
            self._root = tree.getroot()
        return self._root

    @property
    def timedict(self):
        if not hasattr(self, "_timedict"):
            time_slots = self.root.find(
                f".//{self.TIME_ORDER}").findall(f".//{self.TIME_SLOT}")
            self._timedict = {stamp.get(self.TIME_SLOT_ID): int(
                stamp.get(self.TIME_VALUE)) for stamp in time_slots}
        return self._timedict

    def _annotation_to_utterance(self, annotation, tier_id):
        begin = annotation.get(self.TIME_SLOT_REF1)
        end = annotation.get(self.TIME_SLOT_REF2)
        time = [self.timedict[begin], self.timedict[end]]
        return Utterance(utterance=annotation.find(self.ANNOTATION_VALUE).text,
                         time=time,
                         participant=tier_id)

import xml.etree.ElementTree as ET
from ..utterance import Utterance
from .parser import InputFile


class EafFile(InputFile):

    def _extract_metadata(self):
        return {}

    def _extract_utterances(self):
        timedict = self._extract_times(self.root)

        # once again, a terrible loop >.<
        annotations, sorting = [], []

        for tier in self.root.findall(".//TIER"):
            participant = tier.get("TIER_ID")

            for annotation in tier.findall(".//ALIGNABLE_ANNOTATION"):
                utterance = annotation.find("ANNOTATION_VALUE").text

                begin = annotation.get("TIME_SLOT_REF1")
                end = annotation.get("TIME_SLOT_REF2")
                time = [timedict[begin], timedict[end]]

                annotations.append(
                    Utterance(
                        participant=participant,
                        utterance=utterance,
                        time=time
                    ))
                sorting.append(int(begin.replace("ts", "")))

        return [utt for _, utt in sorted(zip(sorting, annotations))]

    @property
    def root(self):
        if not hasattr(self, "_root"):
            tree = ET.parse(self._path)
            self._root = tree.getroot()
        return self._root

    @staticmethod
    def _extract_times(root):
        time_slots = root.find(".//TIME_ORDER").findall(".//TIME_SLOT")
        return {stamp.get("TIME_SLOT_ID"): int(stamp.get("TIME_VALUE")) for stamp in time_slots}

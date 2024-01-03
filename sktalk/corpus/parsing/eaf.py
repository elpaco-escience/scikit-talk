from ..utterance import Utterance
from .parser import InputFile
import xml.etree.ElementTree as ET


class EafFile(InputFile):

    def _extract_metadata(self):
        return {}

    def _extract_utterances(self):
        # once again, a terrible loop >.<
        annotations = []

        for tier in self.root.findall(".//TIER"):
            participant = tier.get("TIER_ID")

            for annotation in tier.findall(".//ALIGNABLE_ANNOTATION"):
                begin = annotation.get("TIME_SLOT_REF1")
                end = annotation.get("TIME_SLOT_REF2")
                utterance = annotation.find("ANNOTATION_VALUE").text

                annotations.append(
                    Utterance(
                        participant=participant,
                        utterance=utterance,
                        time=[begin, end]
                    ))
        return annotations

    @property
    def root(self):
        if not hasattr(self, "_root"):
            tree = ET.parse(self._path)
            self._root = tree.getroot()
        return self._root

from itertools import chain
from pympi.Elan import Eaf
from ..utterance import Utterance
from .parser import InputFile


class EafFile(InputFile):
    """Parser for ELAN files."""

    def _extract_metadata(self):
        return {
            "header": self.pympi_eaf.header,
            "adocument": self.pympi_eaf.adocument,
            "licenses": self.pympi_eaf.licenses,
            "locales": self.pympi_eaf.locales,
            "languages": self.pympi_eaf.languages,
            "media_descriptors": self.pympi_eaf.media_descriptors,
            "properties": self.pympi_eaf.properties,
            "linked_file_descriptors": self.pympi_eaf.linked_file_descriptors,
            "constraints": self.pympi_eaf.constraints,
            "linguistic_types": self.pympi_eaf.linguistic_types,
            "controlled_vocabularies": self.pympi_eaf.controlled_vocabularies,
            "external_refs": self.pympi_eaf.external_refs,
            "lexicon_refs": self.pympi_eaf.lexicon_refs
        }

    def _extract_utterances(self):
        tiers = self.pympi_eaf.tiers
        utterances = [self._annotation_to_utterances(tier) for tier in tiers]
        utterances = list(chain(*utterances))
        sorting = [[*utt.time, index] for index, utt in enumerate(utterances)]
        return [utt for _, utt in sorted(zip(sorting, utterances))]

    @property
    def pympi_eaf(self):
        if not hasattr(self, "_pympi_eaf"):
            self._pympi_eaf = Eaf(self._path)
        return self._pympi_eaf

    def _annotation_to_utterances(self, tier_id):
        data = self.pympi_eaf.get_annotation_data_for_tier(tier_id)
        return [Utterance(annotation[2],
                          time=[annotation[0], annotation[1]],
                          participant=tier_id) for annotation in data]

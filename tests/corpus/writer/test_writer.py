import csv
import json
import os
from contextlib import nullcontext as does_not_raise
import pytest


class TestWriteJson:
    def test_asdict(self, convo):
        """Verify content of dictionary based on conversation"""
        convodict = convo.asdict()
        assert convodict["Utterances"][0] == convo.utterances[0].asdict()
        assert convodict["source"] == convo.metadata["source"]

    @pytest.mark.parametrize("user_path, expected_path", [
        ("tmp_convo.json", "tmp_convo.json"),
        ("tmp_convo", "tmp_convo.json")
    ])
    def test_write_json(self, convo, tmp_path, user_path, expected_path):
        filename = f"{str(tmp_path)}{os.sep}{user_path}"
        convo.write_json(filename)
        filename_exp = f"{str(tmp_path)}{os.sep}{expected_path}"
        assert os.path.exists(filename_exp)
        with open(filename_exp, encoding='utf-8') as f:
            convo_read = json.load(f)
            assert isinstance(convo_read, dict)
            assert convo_read == convo.asdict()


class TestWriteCSV:
    @pytest.mark.parametrize("user_path", [
        ("tmp.csv"),
        ("tmp.json"),
        ("tmp")
    ])
    def test_write_csv_output_paths(self, user_path, convo, tmp_path):
        """Confirm that output paths are set up correctly."""
        filename = f"{str(tmp_path)}{os.sep}{user_path}"
        convo.write_csv(filename)
        metadatapath = f"{str(tmp_path)}{os.sep}tmp_metadata.csv"
        assert os.path.exists(metadatapath)
        utterancepath = f"{str(tmp_path)}{os.sep}tmp_utterances.csv"
        assert os.path.exists(utterancepath)

    def open_csv(self, filename, tmp_path):
        path = f"{str(tmp_path)}{os.sep}{filename}"
        with open(path, 'r', encoding="utf-8") as file:
            reader = csv.reader(file)
            csv_out = list(reader)
        return csv_out

    def test_write_csv_from_conversation(self, convo, convo_meta, expected_csv_metadata, expected_csv_utterances, tmp_path):  # noqa: too-many-arguments
        """Assess the content of the metadata and utterance output csvs"""
        filename = f"{str(tmp_path)}{os.sep}tmp.csv"
        convo.write_csv(filename)
        # metadata should not be updated with this method
        assert convo.metadata == convo_meta

        assert self.open_csv("tmp_metadata.csv",
                             tmp_path) == expected_csv_metadata

        csv_utterances = self.open_csv("tmp_utterances.csv", tmp_path)
        # utterance creation makes additional columns, precise testing is difficult
        csv_utterances = [row[:5] for row in csv_utterances]
        assert csv_utterances == expected_csv_utterances

    def test_write_csv_from_corpus(self, my_corpus_with_convo, expected_csv_metadata_corpus, expected_csv_utterances_corpus, tmp_path):
        """Test writing a corpus to csv"""
        filename = f"{str(tmp_path)}{os.sep}tmp.csv"
        my_corpus_with_convo.write_csv(filename)

        assert self.open_csv("tmp_metadata.csv",
                             tmp_path) == expected_csv_metadata_corpus

        csv_utterances = self.open_csv("tmp_utterances.csv", tmp_path)
        # utterance creation makes additional columns, precise testing is difficult
        csv_utterances = [row[:5] for row in csv_utterances]
        assert csv_utterances == expected_csv_utterances_corpus

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

    @pytest.mark.parametrize("conversation, flags, error_writing", [
        ("convo", (True, True, True), does_not_raise()),
        ("empty_convo", (True, True, False), pytest.warns(
            match="csv is not written")),
        ("convo", (True, False, False), does_not_raise()),
        ("convo", (False, False, False), does_not_raise()),
        ("convo", (False, True, True), does_not_raise()),
    ])
    def test_write_metadata_utterances_optionally(self, conversation, flags, error_writing, tmp_path, request):  # noqa: too-many-arguments
        """Confirm error handling and optional writing of metadata and utterances."""
        conversation = request.getfixturevalue(conversation)
        flag_metadata, flag_utterances, presence_utterances = flags
        filename = f"{str(tmp_path)}{os.sep}tmp.csv"
        with error_writing:
            conversation.write_csv(
                filename, metadata=flag_metadata, utterances=flag_utterances)
        metadatapath = f"{str(tmp_path)}{os.sep}tmp_metadata.csv"
        assert os.path.exists(metadatapath) == flag_metadata
        utterancepath = f"{str(tmp_path)}{os.sep}tmp_utterances.csv"
        assert os.path.exists(utterancepath) == presence_utterances

    def open_csv(self, filename, tmp_path):
        path = f"{str(tmp_path)}{os.sep}{filename}"
        with open(path, 'r', encoding="utf-8") as file:
            reader = csv.reader(file)
            csv_out = list(reader)
        return csv_out

    def test_write_csv_correctly(self, convo, convo_meta, expected_csv_metadata, expected_csv_utterances, tmp_path): # noqa: too-many-arguments
        """Assess the content of the metadata and utterance output csvs"""
        filename = f"{str(tmp_path)}{os.sep}tmp.csv"
        convo.write_csv(filename)
        # metadata should not be updated with this method
        assert convo.metadata == convo_meta

        csv_metadata = self.open_csv("tmp_metadata.csv", tmp_path)
        assert csv_metadata == expected_csv_metadata

        csv_utterances = self.open_csv("tmp_utterances.csv", tmp_path)
        csv_utterances = [row[:5] for row in csv_utterances]
        assert csv_utterances == expected_csv_utterances

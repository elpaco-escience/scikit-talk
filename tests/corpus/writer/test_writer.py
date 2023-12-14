import os
import json
import csv
import pytest
from contextlib import nullcontext as does_not_raise

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
    def open_csv(self, filename, tmp_path):
        path = f"{str(tmp_path)}{os.sep}{filename}"
        with open(path, 'r', encoding="utf-8") as file:
            reader = csv.reader(file)
            csv_out = list(reader)
        return csv_out

    @pytest.mark.parametrize("user_path", [
        ("tmp.csv"),
        ("tmp.json"),
        ("tmp")
    ])
    def test_write_csv(self, user_path, convo, tmp_path):
        """Confirm that output paths are set up correctly."""
        filename = f"{str(tmp_path)}{os.sep}{user_path}"
        convo.write_csv(filename)
        metadatapath = f"{str(tmp_path)}{os.sep}tmp_metadata.csv"
        assert os.path.exists(metadatapath)
        tmp_output_utterances = f"{str(tmp_path)}{os.sep}tmp_utterances.csv"
        assert os.path.exists(tmp_output_utterances)

    def test_write_csv_metadata(self, convo, convo_meta, expected_csv_metadata, tmp_path):
        filename = f"{str(tmp_path)}{os.sep}tmp.csv"
        convo.write_csv(filename)
        # metadata should not be updated with this method
        assert convo.metadata == convo_meta

        csv_metadata = self.open_csv("tmp_metadata.csv", tmp_path)
        assert csv_metadata == expected_csv_metadata


    @pytest.mark.parametrize("conversation, error", [
        ("convo", does_not_raise()),
        ("empty_convo", pytest.raises(FileNotFoundError))
    ])
    def test_write_csv_utterances(self, conversation, error, tmp_path, request):
        conversation = request.getfixturevalue(conversation)

        filename = f"{str(tmp_path)}{os.sep}tmp.csv"
        conversation.write_csv(filename)
        utterancepath = f"{str(tmp_path)}{os.sep}tmp_utterances.csv"

        with error:
            with open(utterancepath, 'r', encoding="utf-8") as file:
                reader = csv.reader(file)
                csv_out = list(reader)

            assert len(csv_out) == len(conversation.utterances)+1
            assert len(set(csv_out[0])) == len(csv_out[0])
            assert csv_out[0][1] == "source"
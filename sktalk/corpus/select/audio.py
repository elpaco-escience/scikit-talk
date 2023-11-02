import abc
import json
import subprocess
import numpy as np


class Audio(abc.ABC):

    @abc.abstractmethod
    def get_audio(self, file_path):
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            file_path
        ]

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        output = json.loads(result.stdout)

        sample_rate = None
        for stream in output["streams"]:
            if stream["codec_type"] == "audio":
                sample_rate = stream["sample_rate"]
                # TODO the channels need to be preserved
                # no_channels = stream["channels"]

        if sample_rate is None:
            raise ValueError("No audio stream found in the file")

        cmd = ["ffmpeg", "-i", file_path, '-f', 's16le',
               '-acodec', 'pcm_s16le',
               '-ar', sample_rate,
               '-ac', '1',
               '-']

        pipe = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              check=True)
        raw_audio = pipe.stdout
        audio_array = np.frombuffer(raw_audio, dtype="int16")
        audio_array = audio_array.astype(np.float32) / np.iinfo(np.int16).max

        return audio_array, int(sample_rate)

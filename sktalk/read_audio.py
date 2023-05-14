import subprocess
import json
import numpy as np

def get_sampling_rate(file_path):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        file_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = json.loads(result.stdout)

    for stream in output["streams"]:
        if stream["codec_type"] == "audio":
            return int(stream["sample_rate"])

    raise ValueError("No audio stream found in the file")

# Replace 'list_audio_3_40_balanced.wav' with the path to your audio file
# file_path = './Elpaco dataset/akhoe_haikom1/state_hospital.wav'

# sampling_rate = get_sampling_rate(file_path)
# print(sampling_rate)


def get_audio_ffmpeg(file_path):
    cmd = ["ffmpeg", "-i", file_path, '-f', 's16le',
           '-acodec', 'pcm_s16le',
           '-ar', '22050',
           '-ac', '1',
           '-']

    pipe = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_audio = pipe.stdout
    audio_array = np.frombuffer(raw_audio, dtype="int16")
    audio_array = audio_array.astype(np.float32) / np.iinfo(np.int16).max

    return audio_array

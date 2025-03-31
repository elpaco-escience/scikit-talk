import librosa
from IPython.display import Audio
from math import floor, ceil
import warnings

## ===== Input/Output =====
def extend_dataframe(df, data_folder = "data"):
    """ Extends our dataframe with:
    - sample rate
    - the audio snippet 
    """
    snippets = subset_all_audios(df, data_folder)
    df.insert(len(df.columns), "audio", snippets)

    rates = samplerate_from_keys(df["key"], data_folder)
    df.insert(len(df.columns), "rate", rates)

    return df

def audio_from_key(key, sr = None, data_folder = "data", **kwargs):
    """ Equivalent to librosa.core.load, but works with keys instead of with filenames """
    try: # This try/catch structure allows the workflow to continue when batch-processing files
        audio, rate = librosa.core.load(filename_from_key(key, data_folder), sr=sr, **kwargs) # sr=None uses the native sampling rate
        audio = audio.astype('float32')
    except:
        warnings.warn(f"Something went wrong with key: {key}")
        audio = [None]
    return audio # We'll ignore the rate in this function output

def samplerate_from_key(key, data_folder = "data", **kwargs):
    """ Equivalent to librosa.get_samplerate, but works with keys instead of with filenames """
    try: # This try/catch structure allows the workflow to continue when batch-processing files
        sr = librosa.get_samplerate(filename_from_key(key, data_folder), **kwargs)
    except:
        warnings.warn(f"Something went wrong with key: {key}")
        sr = 0
    return sr

def subset_audio(audio, start_time, end_time, rate):
    """
    Extracts a subset of the audio signal between the specified start and end times.
    Args:
        audio (list or numpy array): The audio signal to subset.
        start_time (float): The start time in seconds for the subset.
        end_time (float): The end time in seconds for the subset.
        rate (int): The sampling rate of the audio signal.
    Returns:
        list or numpy array: The subset of the audio signal between start_time and end_time.
    Raises:
        Exception: If the start or end indices are out of the bounds of the audio signal.
    """
    start_i = floor(start_time * rate)
    end_i = ceil(end_time * rate)

    try:
        return audio[start_i : end_i]
    except:
        return [None]


def subset_audio_from_key(df, key, row=0, start_time = None, end_time = None, data_folder = "data"):
    """
    Extracts a subset of audio from a given key in the dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe containing audio metadata.
    key (str): The key to identify a file in the dataframe.
    row (int, optional): The row index to use if multiple rows match the key. Defaults to 0.
    start_time (float, optional): The start time for the audio subset. If None, it is taken from the dataframe. Defaults to None.
    end_time (float, optional): The end time for the audio subset. If None, it is taken from the dataframe. Defaults to None.

    Returns:
    np.ndarray: The subset of the audio.
    """

    # Get the audio
    sr = samplerate_from_key(key, data_folder)
    audio = audio_from_key(key, sr, data_folder)

    # Cut it
    ## First, we filter by key
    subdf = df[df.key == key].reset_index() # So the rows start at 0

    ## Because some keys contain multiple rows, we need the logic below
    if(len(subdf) == 1):
        if start_time == None: # If no time is manually provided, it gets it from the dataframe...
            start_time = subdf['start_time']
        if end_time == None: # ... this is useful for testing and prototyping
            end_time = subdf['end_time']
    else:
        if start_time == None: # If no time is manually provided, it gets it from the dataframe...
            start_time = subdf['start_time'][row]
        if end_time == None: # ... this is useful for testing and prototyping
            end_time = subdf['end_time'][row]

    return subset_audio(audio, start_time, end_time, sr)

def subset_all_audios(df, data_folder = "data"):
    """Extracts all the audio snippets

    Args:
        df (pd.Dataframe): our data frame
        data_folder: location of the .wav files

    Returns:
        np.array: A list with the audio clippings
    """
    size = len(df)
    snippets = size * [None] # Pre-allocate an empty list

    counter = 0
    keys = df['key'].unique()
    for key in keys:
        # Open the audio file only once per file (as opposed to once per row)
        audio = audio_from_key(key, data_folder=data_folder)
        rate = samplerate_from_key(key, data_folder)

        # Extract and append the relevant audio snippet
        aux = df[df['key'] == key]
        for i, row in aux.iterrows():
            snippets[counter] = subset_audio(audio, row['start_time'], row['end_time'], rate)
            counter += 1
    
    return snippets

# Some handy list comprehensions
def samplerate_from_keys(keys, data_folder = "data", **kwargs):
    return [samplerate_from_key(key, data_folder, **kwargs) for key in keys]

## ===== Auxiliary functions =====
def filename_from_key(key, data_folder = "data", ext = ".wav"):
    """ Takes the key, returns the filename """
    return data_folder + key + ext #TODO: consider improving this using os.path


def listen_audio_from_key(df, key, row=0, start_time = None, end_time = None, data_folder = "data"):
    """
    Plays a subset of audio from a given key in the dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe containing audio metadata.
    key (str): The key to identify a file in the dataframe.
    row (int, optional): The row index to use if multiple rows match the key. Defaults to 0.
    start_time (float, optional): The start time for the audio subset. If None, it is taken from the dataframe. Defaults to None.
    end_time (float, optional): The end time for the audio subset. If None, it is taken from the dataframe. Defaults to None.
    data_folder (optional): The location of the .wav files

    Returns:
    Audio: A playable audio object
    """
    subset = subset_audio_from_key(df, key, row, start_time, end_time, data_folder)

    return Audio(data = subset, rate = samplerate_from_key(key))

def listen_snippet_from_df(df, row):
    """
    Extracts and returns an audio snippet from a DataFrame, provided it has been appended.

    Args:
        df (pandas.DataFrame): The DataFrame containing audio data and corresponding rates.
        row (int): The index of the row from which to extract the audio snippet.
    Returns:
        Audio: A playable audio object
    """

    return Audio(data = df["audio"][row], rate = df["rate"][row])
# %% libraries and function defs
import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from tqdm import tqdm

from .vad_utils import pad_labels
from .features.short_time_features import extract_temporal_features
from .features.short_time_features import binned_stft


def temporal_features(
    index, file_name, data, label, df,
    frame_size=512, frame_shift=128,
    use_window='hamming', medfilt_size=3
):
    """Extracts short-time energy, magnitude and ZCR of voiced and unvoiced frames.

    Arguments:
        index -- current index, used to create pd dataframe
        file_name -- current .wav filename
        data -- .wav file
        label -- 0-1 labels of current .wav file
        df -- pandas dataframe, where the output will be stored

    Returns:
        time-domain analysis results for voiced and unvoiced frames
    """
    mag, eng, zcr = extract_temporal_features(
        data, use_window,
        frame_size, frame_shift, medfilt_size)

    voice_mag = np.mean(np.where(label == 1, mag, 0))
    unvoice_mag = np.mean(np.where(label == 0, mag, 0))
    voice_eng = np.mean(np.where(label == 1, eng, 0))
    unvoice_eng = np.mean(np.where(label == 0, eng, 0))
    voice_zcr = np.mean(np.where(label == 1, zcr, 0))
    unvoice_zcr = np.mean(np.where(label == 0, zcr, 0))

    current = df.append(pd.DataFrame([[
        file_name,
        voice_mag, unvoice_mag,
        voice_eng, unvoice_eng,
        voice_zcr, unvoice_zcr]], index=[index], columns=df.columns))

    return current


def stft_features(
    index, file_name, data, label, df,
    frame_size=512, frame_shift=128,
    use_window='hamming', sample_rate=16000, bin_mode='coarse'
):
    """Performs STFT on input data.
    Bin each result of STFT into low-freq, med-freq and hi-freq
    and computes energy on each bin for both voiced and unvoiced frames

    Arguments:
        index -- current index, used to create pd dataframe
        file_name -- current .wav filename
        data -- .wav file
        label -- 0-1 labels of current .wav file
        df -- pandas dataframe, where the output will be stored

    Returns:
        freq-domain features of voiced and unvoiced data.
    """
    if bin_mode != 'coarse':
        raise ValueError('Binning mode other than coarse is not supported.')

    bins = binned_stft(
        data,
        use_window, bin_mode,
        frame_size, frame_shift, sample_rate)

    results = [file_name]
    for b in bins:
        results.append(np.mean(np.where(label == 1, b, 0)))
        results.append(np.mean(np.where(label == 0, b, 0)))

    current = df.append(
        pd.DataFrame([results], index=[index], columns=df.columns))

    return current


def naive_feature_analysis(
    path, labels,
    frame_size=512, frame_shift=128,
    use_window='hamming', medfilt_size=3
):
    """Extracts features of a dataset for future analysis.

    Arugments:
        path -- path to dataset
        label -- Dict that maps a .wav file name to its labels.
        It is the return value of read_label_from_file()

    Returns:
        time_analysis -- pandas dataframe of time-domain features
        freq_analysis -- pandas dataframe of freq-domain features
    """
    time_columns = [
        'File',
        'Voiced Magnitude', 'Unvoiced Magnitude',
        'Voiced Energy', 'Unvoiced Energy',
        'Voiced ZCR', 'Unvoiced ZCR'
    ]
    freq_columns = [
        'File',
        'Voiced LowFreq', 'Unvoiced LowFreq',
        'Voiced MedFreq', 'Unvoiced MedFreq',
        'Voiced HighFreq', 'Unvoiced HighFreq'
    ]
    time_analysis = pd.DataFrame(columns=time_columns)
    freq_analysis = pd.DataFrame(columns=freq_columns)

    for root, dirs, files in os.walk(path):
        for index, f in enumerate(tqdm(files)):
            if '.wav' in f:
                rate, raw_data = wavfile.read(os.path.join(path, f))
                data = np.array(raw_data, dtype=float)
                data -= np.mean(data)   # remove dc-offset
                data /= 32767           # normalization

                length = int(np.ceil(
                    (len(data)-(frame_size-frame_shift)) / frame_shift))
                label = pad_labels(labels[f.split('.wav')[0]], length)

                time_analysis = temporal_features(
                    index, f, data, label, time_analysis,
                    frame_size, frame_shift,
                    use_window, medfilt_size)
                freq_analysis = stft_features(
                    index, f, data, label, freq_analysis,
                    frame_size, frame_shift,
                    use_window, rate)

    return time_analysis, freq_analysis

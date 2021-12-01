import librosa
import numpy as np


def extract_mfcc(
        mel_s,
        rate,
        n_mfcc=12,
        use_first_order=False,
        use_third_order=True,):
    """Extracts MFCC features.
    Includes MFCC and 2nd- and 3rd-order deltas.

    Arguments:
        mel_s: 2darray -- Mel-Field Spectrogram.
        rate: int -- Sample rate of input.
        n_mfcc: int -- n_mfcc passed into librosa.feature.mfcc()
        use_first_order: bool -- whether to include 1st-order deltas
        use_third_order: bool -- whether to include 3rd-order deltas

    Returns:
        feature: 1darray -- MFCC and deltas, concatenated into an array.
    """
    mfcc = librosa.feature.mfcc(
        sr=rate,
        S=librosa.core.power_to_db(mel_s),
        n_mfcc=n_mfcc)
    mfcc -= np.mean(mfcc, axis=1).reshape(-1, 1)
    feature = mfcc

    if use_first_order:
        mfcc_d1 = librosa.feature.delta(mfcc, order=1)
        feature = np.concatenate([feature, mfcc_d1], axis=0)

    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    feature = np.concatenate([feature, mfcc_d2], axis=0)

    if use_third_order:
        mfcc_d3 = librosa.feature.delta(mfcc, order=3)
        feature = np.concatenate([feature, mfcc_d3], axis=0)

    return feature


def rms_energy(stft_s):
    """Computes the rms energy of each frame.

    Arguments:
        stft_s: 2darray -- STFT spectrogram.
        rate: int -- Sample rate of input.

    Returns:
        rms: 1darray -- rms energy.
    """
    mag, _ = librosa.core.magphase(stft_s)
    rms = librosa.feature.rms(S=mag)

    return rms


def spectral_feature_extractor(
        data, rate=16000,
        n_frame=512, n_shift=128,
        use_window='hann', n_mfcc=12,
        use_first_order=False, use_third_order=True):
    """Extracts spectral feature.
    Includes mfcc, 1st and 2nd deltas of mfcc, and rms energy.

    Arguments:
        data: 1darray -- input speech signal array
        rate: int -- sample rate
        n_frame: int -- number of samples per frame
        n_shift: int -- number of hops between frames
        use_window: str -- window function to be used in stft
        n_mfcc: int -- n_mfcc passed into librosa.features.mfcc
        use_first_order: bool -- whether to use 1st order deltas
        use_third_order: bool -- whether to use 3rd order deltas

    Returns:
        features: 2darray -- (n_features, n_samples) output array.
    """
    stft = librosa.core.stft(
        data,
        hop_length=n_shift, win_length=n_frame,
        window=use_window)
    mel_s = librosa.feature.melspectrogram(sr=rate, S=np.abs(stft)**2)

    mfcc = extract_mfcc(
        mel_s, rate, n_mfcc=n_mfcc,
        use_first_order=use_first_order, use_third_order=use_third_order)
    rms = rms_energy(stft)

    return np.concatenate([mfcc, rms], axis=0)

# %% libiraries and class and func defs
import os
import sys
import librosa
import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm

from vad.features.short_time_features import short_time_feature_extractor
from vad.features.spectral_features import spectral_feature_extractor
from vad.vad_utils import pad_labels
from vad.vad_utils import read_label_from_file
from vad.evaluate import get_metrics


# always run short_time_analysis.py
# before calling the functions below
def short_time_feature_loader(
    data_set_path,
    label_path,
    use_window='hamming',
    frame_size=512,
    frame_shift=128,
    medfilt_size=3,
    bin_mode='coarse',
):
    """Load and convert all data into a single array of frames and labels.
    Note that the arguments should be EXACTLY THE SAME AS
    the aruguments used in feature analysis.

    Designed for Task 1.

    Returns:
        all_frames -- Array of [N,6], rows are frames and columns are features
        all_labels -- Array of N where each element is a label of a frame
    """
    labels = read_label_from_file(label_path)
    all_frames = np.zeros([0, 6])
    all_labels = np.zeros([0])

    for root, dirs, files in os.walk(data_set_path):
        for index, f in enumerate(tqdm(files)):
            if '.wav' in f:
                rate, raw_data = wavfile.read(os.path.join(data_set_path, f))
                data = np.array(raw_data, dtype=float)
                data -= np.mean(data)   # remove dc-offset
                data /= 32767           # normalize

                frames = short_time_feature_extractor(
                    data,
                    use_window,
                    frame_size,
                    frame_shift,
                    medfilt_size,
                    bin_mode,
                    rate
                ).T
                ground_truth = pad_labels(
                    labels[f.split('.wav')[0]], frames.shape[0])
                all_frames = np.concatenate([all_frames, frames], axis=0)
                all_labels = np.concatenate([all_labels, ground_truth], axis=0)

    return all_frames, all_labels


def quick_pass(classifier, frames, labels):
    """If the data and labels are ready
    this function can be called to evaluate the model directly

    Returns:
        auc, err -- metrics of current classifier on the current dataset
    """
    pred = classifier.predict(frames)
    auc, eer = get_metrics(pred, labels)
    return auc, eer


def spectral_feature_loader(
    data_set_path,
    label_path,
    use_window='hann',
    frame_size=512,
    frame_shift=128,
    n_mfcc=12,
    use_first_order=False,
    use_third_order=True,
):
    """Load training (or evaluating) data from a given directory.
    Designed for Task 2.

    Arguments:
        data_set_path: str -- path to dataset.
        label_path: str -- path to labels of corresponding dataset.
        use_window: str -- window funtion to be used.
        frame_size: int -- number of samples per frame.
        frame_shift: int -- number of hops between frames.
        n_mfcc: int -- n_mfcc passed into librosa.features.mfcc()
        use_first_order: bool -- whether to use 1st order deltas
        use_third_order: bool -- whether to use 3rd order deltas

    Returns:
        X_train: 2darray -- (n_features, n_samples) array.
        sample_lengths: 2darray -- (n_samples) array,
            the length of each sample.
            Useful when an GMMHMM model is used for VAD.
        Y_train: 1darray -- label of each sample.
    """
    print_loader_info(
        data_set_path,
        use_window,
        frame_size,
        frame_shift,
        n_mfcc,
        use_first_order,
        use_third_order,)
    labels = read_label_from_file(label_path)
    mfcc_batches = 2
    if use_first_order:
        mfcc_batches += 1
    if use_third_order:
        mfcc_batches += 1
    n_features = mfcc_batches * n_mfcc + 1
    X_train = np.zeros([n_features, 0])
    sample_lengths = np.zeros(0, dtype=int)
    Y_train = np.zeros([0])
    for root, dirs, files in os.walk(data_set_path):
        t = tqdm(files)
        for f in t:
            if '.wav' in f:
                t.set_description(
                    'Current File {:s}'.format(f))
                t.refresh()
                data, rate = librosa.core.load(
                    os.path.join(data_set_path, f), sr=None)
                data -= np.mean(data)

                file_feature = spectral_feature_extractor(
                    data, rate, frame_size, frame_shift, use_window, n_mfcc,
                    use_first_order=use_first_order,
                    use_third_order=use_third_order)
                sample_lengths = np.append(
                    sample_lengths,
                    file_feature.shape[-1])

                X_train = np.concatenate([X_train, file_feature], axis=-1)
                ground_truth = pad_labels(
                    labels[f.split('.wav')[0]], file_feature.shape[-1])
                Y_train = np.concatenate([Y_train, ground_truth])

    return X_train, sample_lengths, Y_train


def print_loader_info(
        data_set_path,
        use_window,
        frame_size,
        frame_shift,
        n_mfcc,
        use_first_order,
        use_third_order,):
    print(
        '[DataLoader] Loading data from {:s}'.format(data_set_path),
        file=sys.stderr)
    print(
        '===================================',
        file=sys.stderr)
    print(
        f'frame_size: {frame_size} | frame_shift: {frame_shift}',
        file=sys.stderr)
    print(
        f'use_window: {use_window} | n_mfcc: {n_mfcc}',
        file=sys.stderr)
    print(
        f'first_order: {use_first_order} | third_order: {use_third_order}',
        file=sys.stderr)

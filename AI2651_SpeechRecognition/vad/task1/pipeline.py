import os
import numpy as np
from tqdm import tqdm
import sys
import scipy.io.wavfile as wavfile
from ..classifiers.basic import BasicThresholdClassifer
from ..data_loader import short_time_feature_loader
from ..features.short_time_features import short_time_feature_extractor
from ..vad_utils import prediction_to_vad_label
from ..evaluate import get_metrics


def run_on_dev(
        classifier: BasicThresholdClassifer,
        dev_set_path,
        dev_label_path,
        medfilt_size=15,
        n_frame=512,
        n_shift=128,):
    print('Running Task 1 on dev set', file=sys.stderr)

    frames, labels = short_time_feature_loader(
        data_set_path=dev_set_path,
        label_path=dev_label_path,
        use_window='hamming',
        frame_size=n_frame,
        frame_shift=n_shift,
        medfilt_size=medfilt_size,
        bin_mode='coarse'
    )
    pred = classifier.predict(frames)

    auc, eer = get_metrics(pred, labels)
    print('Run Finished.')
    print('  - AUC: {:.4f}'.format(auc))
    print('  - EER: {:.4f}'.format(eer))


def run_on_test(
        classifier: BasicThresholdClassifer,
        test_set_path,
        medfilt_size=15,
        n_frame=512,
        n_shift=128,):
    print('Running Task 1 on test set', file=sys.stderr)
    with open('./vad/task1/test_label_task1.txt', 'w') as output:
        # load data from test set
        for root, dirs, files in os.walk(test_set_path):
            for f in tqdm(files):
                if '.wav' in f:
                    output.write(f.replace('.wav', ' '))
                    rate, data = wavfile.read(os.path.join(test_set_path, f))
                    data = np.array(data, dtype=float)
                    data -= np.mean(data)   # remove dc-offset
                    data /= 32767           # normalize

                    # feature extraction
                    frames = short_time_feature_extractor(
                        data, medfilt_size=medfilt_size,
                        frame_size=n_frame, frame_shift=n_shift).T
                    # predict labels
                    pred = classifier.predict(frames)
                    result = prediction_to_vad_label(pred)
                    output.write(result)
                    output.write('\n')

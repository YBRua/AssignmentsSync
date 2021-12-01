import os
import sys
import pickle
from vad.vad_utils import prediction_to_vad_label
from vad.features.spectral_features import spectral_feature_extractor
import librosa
import numpy as np
from tqdm import tqdm
from ..data_loader import print_loader_info, spectral_feature_loader
from ..evaluate import get_metrics
from ..classifiers.dualGMM import DualGMMClassifier


def train(
        VADClassifier: DualGMMClassifier,
        train_set_path,
        train_label_path,
        use_window='hann',
        n_frame=512,
        n_shift=128,
        n_mfcc=12,
        use_first_order=False,
        use_third_order=True):
    # load datasets
    print('Loading training set...', file=sys.stderr)
    if os.path.exists('./data/training_set.pkl'):
        X_train, sample_lengths, Y_train = pickle.load(
            open('./data/training_set.pkl', 'rb'))
    else:
        X_train, sample_lengths, Y_train = spectral_feature_loader(
            train_set_path, train_label_path,
            use_window=use_window,
            frame_size=n_frame, frame_shift=n_shift,
            n_mfcc=n_mfcc,
            use_first_order=use_first_order, use_third_order=use_third_order,)
        pickle.dump(
            [X_train, sample_lengths, Y_train],
            open('./data/training_set.pkl', 'wb'))

    # convert data
    X_train = X_train.T  # (n_features, n_samples) -> (n_samples, n_features)

    # train the model
    print('Training model...', file=sys.stderr)
    VADClassifier.fit(X_train, Y_train)

    # evaluate on training set
    pred_train_prob = VADClassifier.predict_smoothed_proba(X_train)
    pred_train = np.where(pred_train_prob >= 0.5, 1, 0)

    auc_prob, eer_prob = get_metrics(pred_train_prob, Y_train)
    auc, eer = get_metrics(pred_train, Y_train)

    print('Done.')
    print(
        '  - AUC on train: {:.4f} | {:.4f}'.format(auc, auc_prob))
    print(
        '  - EER on train: {:.4f} | {:.4f}'.format(eer, eer_prob))

    return VADClassifier, auc, eer


def evaluate(
        VADClassifier: DualGMMClassifier,
        dev_set_path,
        dev_label_path,
        use_window='hann',
        n_frame=512,
        n_shift=128,
        n_mfcc=12,
        use_first_order=False,
        use_third_order=True):
    print('Loading dev set...', file=sys.stderr)
    if os.path.exists('./data/dev_set.pkl'):
        X_dev, sample_lengths, Y_dev = pickle.load(
            open('./data/dev_set.pkl', 'rb'))
    else:
        X_dev, sample_lengths, Y_dev = spectral_feature_loader(
            dev_set_path, dev_label_path,
            use_window=use_window,
            frame_size=n_frame, frame_shift=n_shift,
            n_mfcc=n_mfcc,
            use_first_order=use_first_order, use_third_order=use_third_order,)
        pickle.dump(
            [X_dev, sample_lengths, Y_dev],
            open('./data/dev_set.pkl', 'wb'))

    X_dev = X_dev.T

    print('Evaluating model...', file=sys.stderr)
    pred_dev_prob = VADClassifier.predict_smoothed_proba(X_dev)
    pred_dev = np.where(pred_dev_prob >= 0.5, 1, 0)

    auc_prob, eer_prob = get_metrics(pred_dev_prob, Y_dev)
    auc, eer = get_metrics(pred_dev, Y_dev)

    print('Done')
    print(
        '  - AUC on dev:   {:.4f} | {:.4f}'.format(auc, auc_prob))
    print(
        '  - EER on dev:   {:.4f} | {:.4f}'.format(eer, eer_prob))

    return auc, eer


def run_on_test(
        VADClassifier: DualGMMClassifier,
        test_set_path,
        use_window='hann',
        n_frame=512,
        n_shift=128,
        n_mfcc=12,
        use_first_order=False,
        use_third_order=True,):
    print('Running on test set...', file=sys.stderr)
    print_loader_info(
        test_set_path, use_window,
        n_frame, n_shift, n_mfcc,
        use_first_order, use_third_order)

    with open('./vad/task2/test_label_task2.txt', 'w') as output:
        # load data
        for root, dirs, files in os.walk(test_set_path):
            t = tqdm(files)
            for f in t:
                if '.wav' in f:
                    t.set_description(
                        'Current File {:s}'.format(f))
                    t.refresh()
                    output.write(f.replace('.wav', ' '))
                    data, rate = librosa.core.load(
                        os.path.join(test_set_path, f), sr=None)
                    data -= np.mean(data)

                    file_feature = spectral_feature_extractor(
                        data, rate, n_frame, n_shift,
                        use_window=use_window, n_mfcc=n_mfcc,
                        use_first_order=use_first_order,
                        use_third_order=use_third_order,)

                    # predict
                    pred = VADClassifier.predict_smoothed_proba(file_feature.T)
                    result = prediction_to_vad_label(pred)
                    output.write(result)
                    output.write('\n')

    print('Done!', file=sys.stderr)

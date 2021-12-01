import os
import pickle
import vad.task2.pipeline as task2Ppl
from vad.classifiers.dualGMM import DualGMMClassifier
from vad.data_loader import spectral_feature_loader

train_set_path = './wavs/train'
train_label_path = './data/train_label.txt'
dev_set_path = './wavs/dev'
dev_label_path = './data/dev_label.txt'

n_frame = 512
n_shift = 128
n_mfcc = 12

n_mixture_list = [1, 2, 3, 5, 7]

if os.path.exists('./data/training_set.pkl'):
    X_train, sample_lengths, Y_train = pickle.load(
        open('./data/training_set.pkl', 'rb'))
else:
    X_train, sample_lengths, Y_train = spectral_feature_loader(
        train_set_path, train_label_path,
        frame_size=n_frame, frame_shift=n_shift,
        n_mfcc=n_mfcc, use_first_order=False, use_third_order=True)
    pickle.dump(
        [X_train, sample_lengths, Y_train],
        open('./data/training_set.pkl', 'wb'))

if os.path.exists('./data/dev_set.pkl'):
    X_dev, sample_lengths, Y_dev = pickle.load(
        open('./data/dev_set.pkl', 'rb'))
else:
    X_dev, sample_lengths, Y_dev = spectral_feature_loader(
        dev_set_path, dev_label_path,
        frame_size=n_frame, frame_shift=n_shift,
        n_mfcc=n_mfcc, use_first_order=False, use_third_order=True)
    pickle.dump(
        [X_dev, sample_lengths, Y_dev],
        open('./data/dev_set.pkl', 'wb'))

for n_mixture in n_mixture_list:
    VADClassifier = DualGMMClassifier(
        n_components=n_mixture,
        covariance_type='full',
        max_iter=500,
        verbose=1,
        random_state=1919810,
    )

    task2Ppl.train(VADClassifier, train_set_path, train_label_path)
    task2Ppl.evaluate(VADClassifier, dev_set_path, dev_label_path)

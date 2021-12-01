import os
import pickle
import numpy as np
from vad.data_loader import spectral_feature_loader
from vad.classifiers.dualGMM import DualGMMClassifier
from vad.evaluate import get_metrics

train_set_path = './wavs/train'
dev_set_path = './wavs/dev'
train_label_path = './data/train_label.txt'
dev_label_path = './data/dev_label.txt'
parent_path = './vad/task2/feature_engineering/'

n_mfcc_list = [8, 12, 15, 20]

# all deltas
for n_mfcc in n_mfcc_list:
    train_name = ''.join(['train_123_', str(n_mfcc), '.pkl'])
    if not os.path.exists(os.path.join(parent_path, train_name)):
        X_train, sample_lengths, Y_train = spectral_feature_loader(
            train_set_path, train_label_path,
            n_mfcc=n_mfcc,
            use_first_order=True, use_third_order=True)

        pickle.dump(
            [X_train, sample_lengths, Y_train],
            open(os.path.join(parent_path, train_name), 'wb'))
    else:
        X_train, _, Y_train = pickle.load(
            open(os.path.join(parent_path, train_name), 'rb'))

    dev_name = ''.join(['dev_123_', str(n_mfcc), '.pkl'])
    if not os.path.exists(os.path.join(parent_path, dev_name)):
        X_dev, sample_lengths_dev, Y_dev = spectral_feature_loader(
            dev_set_path, dev_label_path,
            n_mfcc=n_mfcc,
            use_first_order=True, use_third_order=True)

        pickle.dump(
            [X_dev, sample_lengths_dev, Y_dev],
            open(os.path.join(parent_path, dev_name), 'wb'))
    else:
        X_dev, _, Y_dev = pickle.load(
            open(os.path.join(parent_path, dev_name), 'rb'))

    X_mfcc = X_train[0:n_mfcc, :]
    X_mfcc_d1 = X_train[n_mfcc:2*n_mfcc, :]
    X_mfcc_d2 = X_train[2*n_mfcc:3*n_mfcc, :]
    X_mfcc_d3 = X_train[3*n_mfcc:4*n_mfcc, :]
    X_rms = np.expand_dims(X_train[-1, :], axis=0)

    dev_mfcc = X_dev[0:n_mfcc, :]
    dev_mfcc_d1 = X_dev[n_mfcc:2*n_mfcc, :]
    dev_mfcc_d2 = X_dev[2*n_mfcc:3*n_mfcc, :]
    dev_mfcc_d3 = X_dev[3*n_mfcc:4*n_mfcc, :]
    dev_rms = np.expand_dims(X_dev[-1, :], axis=0)

    X_train_12 = np.concatenate(
        [X_mfcc, X_mfcc_d1, X_mfcc_d2, X_rms]).T
    X_train_23 = np.concatenate(
        [X_mfcc, X_mfcc_d2, X_mfcc_d3, X_rms]).T

    X_dev_12 = np.concatenate(
        [dev_mfcc, dev_mfcc_d1, dev_mfcc_d2, dev_rms]).T
    X_dev_23 = np.concatenate(
        [dev_mfcc, dev_mfcc_d2, dev_mfcc_d3, dev_rms]).T

    clf = DualGMMClassifier(
        n_components=3,
        verbose=1,
        random_state=1919810,
    )
    clf.fit(X_train_12, Y_train)
    pred = clf.predict(X_dev_12)
    auc, eer = get_metrics(pred, Y_dev)
    print("MFCC: ", n_mfcc)
    print('Order 1, 2')
    print('AUC: {:.4f} | EER" {:.4f}'.format(auc, eer))
    print()

    clf = DualGMMClassifier(
        n_components=3,
        verbose=1,
        random_state=1919810,
    )
    clf.fit(X_train_23, Y_train)
    pred = clf.predict(X_dev_23)
    auc, eer = get_metrics(pred, Y_dev)
    print("MFCC: ", n_mfcc)
    print('Order 2, 3')
    print('AUC: {:.4f} | EER" {:.4f}'.format(auc, eer))
    print()

    clf = DualGMMClassifier(
        n_components=3,
        verbose=1,
        random_state=1919810,
    )
    clf.fit(X_train.T, Y_train)
    pred = clf.predict(X_dev.T)
    auc, eer = get_metrics(pred, Y_dev)
    print("MFCC: ", n_mfcc)
    print('Order 1, 2, 3')
    print('AUC: {:.4f} | EER" {:.4f}'.format(auc, eer))
    print()

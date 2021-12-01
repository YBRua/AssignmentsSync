# %%
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from vad.classifiers.dualGMM import DualGMMClassifier
import os
import sys
import librosa
import numpy as np
import pickle as pkl
from tqdm import tqdm

from vad.features.spectral_features import spectral_feature_extractor
from vad.vad_utils import read_label_from_file, pad_labels
from vad.evaluate import get_metrics

FRAME_SIZE = 0.032   # 100ms per frame
FRAME_SHIFT = 0.008  # 40ms frame-shift
SAMPLE_RATE = 16000  # 16kHz sample rate
N_FRAME = int(FRAME_SIZE * SAMPLE_RATE)
N_SHIFT = int(FRAME_SHIFT * SAMPLE_RATE)

# set file paths here
train_set_path = './wavs/dev'
dev_label_path = './data/dev_label.txt'

# %%
X_train = np.zeros([37, 0])
Y_train = np.zeros([0])
labels = read_label_from_file(dev_label_path)
sample_lengths = np.zeros(0, dtype=int)
print('Loading Data...', file=sys.stderr)
for root, dirs, files in os.walk(train_set_path):
    i = 0
    for f in tqdm(files):
        if '.wav' in f:
            data, rate = librosa.core.load(
                os.path.join(train_set_path, f), sr=None)
            data -= np.mean(data)

            wav_feature = spectral_feature_extractor(
                data, 16000, N_FRAME, N_SHIFT, n_mfcc=12)
            sample_lengths = np.append(sample_lengths, wav_feature.shape[-1])

            X_train = np.concatenate([X_train, wav_feature], axis=-1)
            ground_truth = pad_labels(
                labels[f.split('.wav')[0]], wav_feature.shape[-1])
            Y_train = np.concatenate([Y_train, ground_truth])

# %%
pkl.dump([X_train, sample_lengths, Y_train], open('training_data.pkl', 'wb'))

# %%

X_train, sample_lengths, Y_train = pkl.load(open('training_data.pkl', 'rb'))
X_train = X_train.T

VADClassifier = DualGMMClassifier(n_components=2)
voiced_gmm = GaussianMixture(n_components=2)

# %%
VADClassifier = VADClassifier.fit(X_train, Y_train)
# %%
pred = VADClassifier.predict_proba(X_train)
# %%
Y_pred = pred[:, 0]
# Y_pred = np.where(Y_pred > 0.5, 1, 0)
auc, eer = get_metrics(Y_pred, Y_train)
print('Run Finished.')
print('  - AUC: {:.4f}'.format(auc))
print('  - EER: {:.4f}'.format(eer))

# %%


mu = VADClassifier.unvoiced_gmm.means_[4]
sigma = VADClassifier.unvoiced_gmm.covariances_[4]
print(multivariate_normal(mu, sigma).pdf(X_train[0, :]))

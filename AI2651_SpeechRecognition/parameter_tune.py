# %% import libraries and initialize
import os
import sys
import pickle
from tqdm import trange

from vad.classifiers.basic import BasicThresholdClassifer
from vad.data_loader import quick_pass, short_time_feature_loader
from vad.short_time_analysis import naive_feature_analysis
from vad.vad_utils import read_label_from_file

if __name__ == '__main__':
    FRAME_SIZE = 0.032   # 100ms per frame
    FRAME_SHIFT = 0.008  # 40ms frame-shift
    SAMPLE_RATE = 16000  # 16kHz sample rate
    N_FRAME = int(FRAME_SIZE * SAMPLE_RATE)
    N_SHIFT = int(FRAME_SHIFT * SAMPLE_RATE)

    dev_set_path = './wavs/dev'
    dev_label_path = './data/dev_label.txt'

    med_filtering = 15
    max_epoch = 10
    max_iter = 500

    # baseline: default params
    # auc: 0.9051
    # eer: 0.1508
    baseline_auc = 0.9051
    baseline_eer = 0.1508
    highest = baseline_auc
    results = []

    labels = read_label_from_file()
    time, freq = naive_feature_analysis(
        dev_set_path, labels,
        N_FRAME, N_SHIFT,
        medfilt_size=med_filtering
    )

    time.to_csv('./time_domain_features.csv', index=False)
    freq.to_csv('./freq_domain_features.csv', index=False)

    classifier = BasicThresholdClassifer(time, freq)

    print('Loading data...', file=sys.stderr)
    if os.path.exists('./dataset.pkl'):
        frames, truths = pickle.load(open('./dataset.pkl', 'rb'))
    else:
        frames, truths = short_time_feature_loader(
            dev_set_path,
            dev_label_path,
            medfilt_size=med_filtering)
        pickle.dump([frames, truths], open('./dataset.pkl', 'wb'))

    print('Running parameter tuning...', file=sys.stderr)
    for e in range(max_epoch):
        classifier = BasicThresholdClassifer(time, freq)
        t = trange(max_iter, desc='Highest:', leave=True)
        for i in t:
            auc, eer = quick_pass(classifier, frames, truths)
            t.set_description(
                'Current: {:.4f} || Highest: {:.4f}'.format(auc, highest)
            )
            t.refresh()
            if auc > highest:
                highest = auc
                results.append([classifier.weight, auc, eer])
            # add random pertubation
            # hopefully the classifier will work better
            classifier.random_update_params()

    pickle.dump(results, open('./tune_results.pkl', 'wb'))

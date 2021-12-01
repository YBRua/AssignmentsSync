import os
import pandas as pd
from vad.short_time_analysis import naive_feature_analysis
from vad.vad_utils import read_label_from_file
from vad.classifiers.basic import BasicThresholdClassifer, ScoreWeight
import vad.task1.pipeline as task1Ppl

dev_set_path = './wavs/dev'
dev_label_path = './data/dev_label.txt'
test_set_path = './wavs/test'
n_frame = 512
n_shift = 128
medfilt_size = 15

optimal_weight = ScoreWeight(
    2.3660515227, 2.20055434, 0.4418205904,
    0.3016455199, 4.3315168325, 1.7074504699,
    3.4955110421, 5.6886246655, 6.6389220191
)

labels = read_label_from_file(dev_label_path)
if (not os.path.exists('./vad/task1/time_domain_features.csv')
        or not os.path.exists('./vad/task1/freq_domain_features.csv')):
    time, freq = naive_feature_analysis(
        dev_set_path, labels,
        n_frame, n_shift,
        medfilt_size=medfilt_size)

    time.to_csv('./vad/task1/time_domain_features.csv', index=False)
    freq.to_csv('./vad/task1/freq_domain_features.csv', index=False)
else:
    time = pd.read_csv('./vad/task1/time_domain_features.csv')
    freq = pd.read_csv('./vad/task1/freq_domain_features.csv')

classifier = BasicThresholdClassifer(time, freq, optimal_weight)
task1Ppl.run_on_dev(classifier, dev_set_path, dev_label_path)
task1Ppl.run_on_test(classifier, test_set_path)

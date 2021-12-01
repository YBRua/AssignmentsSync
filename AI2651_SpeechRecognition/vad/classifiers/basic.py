from collections import namedtuple
import numpy as np


Stats = namedtuple(
    'Stats',
    [
        'magnitude',
        'energy',
        'zcr',
        'lowfreq',
        'medfreq',
        'highfreq',
    ]
)


ScoreWeight = namedtuple(
    'ScoreWeight',
    [
        'mag', 'enr', 'zcr',
        'low', 'med', 'high',
        'state_weight',
        'primary_passes', 'secondary_passes'
    ]
)


class BasicThresholdClassifer():
    def __init__(self, time, freq, score_weight=None):
        self.voiced = Stats(
            time['Voiced Magnitude'].min(),
            time['Voiced Energy'].min(),
            time['Voiced ZCR'].min(),
            freq['Voiced LowFreq'].mean(),
            freq['Voiced MedFreq'].min(),
            freq['Voiced HighFreq'].min()
        )
        self.unvoiced = Stats(
            time['Unvoiced Magnitude'].max(),
            time['Unvoiced Energy'].max(),
            time['Unvoiced ZCR'].max(),
            freq['Unvoiced LowFreq'].max(),
            freq['Unvoiced MedFreq'].max(),
            freq['Unvoiced HighFreq'].max()
        )

        self.mag_boundary =\
            (self.voiced.magnitude + self.unvoiced.magnitude) / 2
        self.energy_boundary =\
            (self.voiced.energy + self.unvoiced.energy) / 2
        self.zcr_boundary =\
            (self.voiced.zcr + self.unvoiced.zcr) / 2
        self.lowfreq_boundary =\
            (self.voiced.lowfreq + self.unvoiced.lowfreq) / 2
        self.medfreq_boundary =\
            (self.voiced.medfreq + self.unvoiced.medfreq) / 2
        self.highfreq_boundary =\
            (self.voiced.highfreq + self.unvoiced.highfreq) / 2

        self.state = 0

        if score_weight is None:
            self.weight = ScoreWeight(
                mag=2,
                enr=2,
                zcr=0,
                low=1,
                med=4,
                high=1,
                state_weight=2,
                primary_passes=5,
                secondary_passes=6
            )
        else:
            self.weight = score_weight

    def _check_primary_features(self, x):
        passes = 0
        if x[0] > self.mag_boundary:
            passes += self.weight.mag
        if x[1] > self.energy_boundary:
            passes += self.weight.enr
        if x[4] > self.medfreq_boundary:
            passes += self.weight.med

        return passes + self.state * self.weight.state_weight

    def _check_secondary_features(self, x, passes):
        if x[2] > self.zcr_boundary:
            passes += self.weight.zcr
        if x[3] > self.lowfreq_boundary:
            passes += self.weight.low
        if x[5] > self.highfreq_boundary:
            passes += self.weight.high
        return passes

    def random_update_params(self):
        param_list = list(self.weight)
        for i in range(len(param_list)):
            pertub = np.random.randn() * 0.1
            if pertub + param_list[i] >= 0:
                param_list[i] += pertub
        self.weight = ScoreWeight(*param_list)

    def pred_one_frame(self, x):
        primal_passes = self._check_primary_features(x)
        secondary_passes = self._check_secondary_features(x, primal_passes)
        if primal_passes >= self.weight.primary_passes:
            self.state = 1
            return 1
        elif secondary_passes >= self.weight.secondary_passes:
            self.state = 1
            return 1
        else:
            self.state = 0
            return 0

    def predict(self, x):
        pred = []
        for frame in x:
            pred.append(self.pred_one_frame(frame))

        return pred

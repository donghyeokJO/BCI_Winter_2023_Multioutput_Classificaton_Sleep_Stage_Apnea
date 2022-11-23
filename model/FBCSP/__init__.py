import numpy as np

from mne.decoding import CSP
from model.util import butter_bandpass_filter
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class FBCSP(object):
    def __init__(self, sampling_rate, n_components, n_select, **kwargs):
        self.sampling_rate = sampling_rate
        self.select_k = SelectKBest
        self.scaler = StandardScaler()
        self.classifier = LinearDiscriminantAnalysis()
        self.low_cut, self.high_cut, self.interval = 4, 40, 4
        self.bands = np.arange(self.low_cut, self.high_cut, self.interval)
        self.csp_list = {
            '{}-{}'.format(band, band + self.interval):
                CSP(n_components=n_components) for band in self.bands
        }

    def temporal_spatial_filtering(self, x, y=None, train: bool = False):
        new_x = []

        for band in self.bands:
            start_band, end_band = band, band + self.interval

            # stage 1 : temporal filtering
            x_filter = butter_bandpass_filter(data=x, lowcut=start_band, highcut=end_band, fs=self.sampling_rate)

            # stage 2 : spectral filtering
            if train:
                csp = self.csp_list['{}-{}'.format(start_band, end_band)]
                x_filter = csp.fit_transform(x_filter, y)
                self.csp_list['{}-{}'.format(start_band, end_band)] = csp

            else:
                csp = self.csp_list['{}-{}'.format(start_band, end_band)]
                x_filter = csp.transform(x_filter)

            new_x.append(x_filter)

        new_x = np.concatenate(new_x, axis=1)
        return new_x

    def train_(self, x, y, *args):
        # stage 1,2 => acquire and combine features of different frequency bands
        x = self.temporal_spatial_filtering(x, y, train=True)

        # stage 3 => band selection => get the best k features base on mutual information algorithm
        x = self.select_k.fit_transform(x, y)

        # stage 4 => classification
        x = self.scaler.fit_transform(x)
        self.classifier.fit(x, y)

    def predicate(self, x):
        # stage 1, 2 => acquire and combine features of different frequency bands
        x = self.temporal_spatial_filtering(x, train=False)

        # stage 3 => band selection => get the best k features base on mutual information algorithm
        x = self.select_k.transform(x)

        # stage 4: classification
        x = self.scaler.transform(x)
        out, prob = self.classifier.predict(x), self.classifier.predict_proba(x)

        return out, prob

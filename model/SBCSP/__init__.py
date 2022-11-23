import numpy as np

from model.util import butter_bandpass_filter
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class SBCSP(object):
    # Sub Band Common Spectral Pattern
    def __init__(self, sampling_rate, n_components, **kwargs):
        self.sampling_rate = sampling_rate
        self.low_cut, self.high_cut, self.interval = 4, 40, 4
        self.bands = np.arange(self.low_cut, self.high_cut, self.interval)
        self.csp_list = {
            '{}-{}'.format(band, band+self.interval):
                CSP(n_components=n_components) for band in self.bands
        }
        self.lda_list = {
            '{}-{}'.format(band, band+self.interval):
                LinearDiscriminantAnalysis() for band in self.bands
        }
        self.classifier = LinearDiscriminantAnalysis()

    def train_(self, x, y, *args):
        # [stage 1] => common spectral pattern
        x = self.common_spectral_pattern(x, y, train=True)

        # [stage 2] => linear discriminant analysis
        x = self.linear_discriminant_analysis(x, y, train=True)

        # [stage 3] : classification
        self.classifier.fit(x, y)

    def predicate(self, x):
        # [stage 1] => common spectral pattern
        x = self.common_spectral_pattern(x, train=False)

        # [stage 2] => linear discriminant analysis
        x = self.linear_discriminant_analysis(x, train=False)

        # [stage 3] => classification
        out, prob = self.classifier.predict(x), self.classifier.predict_proba(x)
        return out, prob

    def predicate_prob(self, x):
        # [stage 1] => common spectral pattern
        x = self.common_spectral_pattern(x, train=False)

        # [stage 2] => linear discriminant analysis
        x = self.linear_discriminant_analysis(x, train=False)

        # [stage 3] => classification
        prob = self.classifier.predict_proba(x)
        return prob

    def common_spectral_pattern(self, x, y=None, train=True):
        new_x = []
        for band in self.bands:
            start_band, end_band = band, band + self.interval
            # [step 1] : band pass filter
            x_filter = butter_bandpass_filter(data=x, lowcut=start_band, highcut=end_band, fs=self.sampling_rate)
            # [step 2] : common spectral pattern
            if train:
                csp = self.csp_list['{}-{}'.format(start_band, end_band)]
                x_filter = csp.fit_transform(x_filter, y)
                self.csp_list['{}-{}'.format(start_band, end_band)] = csp
            else:
                csp = self.csp_list['{}-{}'.format(start_band, end_band)]
                x_filter = csp.transform(x_filter)

            new_x.append(x_filter)
        return new_x

    def linear_discriminant_analysis(self, x, y=None, train=True):
        new_x = []
        for i, band in enumerate(self.bands):
            x_sample = x[i]
            start_band, end_band = band, band + self.interval
            if train:
                lda = self.lda_list['{}-{}'.format(start_band, end_band)]
                x_filter = lda.fit_transform(x_sample, y)
                self.lda_list['{}-{}'.format(start_band, end_band)] = lda
                new_x.append(x_filter)
            else:
                lda = self.lda_list['{}-{}'.format(start_band, end_band)]
                x_filter = lda.transform(x_sample)
                new_x.append(x_filter)

        new_x = np.concatenate(new_x, axis=-1)
        return new_x

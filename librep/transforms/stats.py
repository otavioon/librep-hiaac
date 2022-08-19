import numpy as np
from scipy import stats
from scipy.signal import find_peaks

from librep.config.type_definitions import ArrayLike
from librep.base.transform import Transform


class StatsTransform(Transform):
    def __init__(self, keep_values: bool = False, capture_statistical: bool = True, capture_indices: bool = True):
        self.keep_values = keep_values
        self.capture_statistical = capture_statistical
        self.capture_indices = capture_indices

    def fit(self, X, y):
        pass

    def transform(self, X: ArrayLike):
        values = []
        if self.capture_statistical:
            values += [
                np.mean(X),
                np.std(X),
                np.mean(np.absolute(X - np.mean(X))),
                np.min(X),
                np.max(X),
                np.max(X)-np.min(X),
                np.median(X),
                np.median(np.absolute(X - np.median(X))),
                np.percentile(X, 75) - np.percentile(X, 25),
                np.sum(X < 0),
                np.sum(X > 0),
                np.sum(X > np.mean(X)),
                len(find_peaks(X)[0]),
                stats.skew(X),
                stats.kurtosis(X),
                np.sum(X**2)/100,
            ]
        if self.capture_indices:
            values += [
                np.argmax(X),
                np.argmin(X),
                np.argmax(X)-np.argmin(X)
            ]

        values = np.array(values)
        if self.keep_values:
            return np.concatenate([X, values])
        else:
            return values

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

from librep.config.type_definitions import ArrayLike
from librep.base.transform import Transform


class StatsTransform(Transform):
    """Extract statistical information of a sample.

    Parameters
    ----------
    keep_values : bool
        If true, the statistical information is concatenated with the input
        sample. (the default is False).
    capture_statistical : bool
        If True, extract statistical information about the sample.
    capture_indices : bool
        If True, extract statistical information about the indexes of the sample.

    """

    def __init__(self, keep_values: bool = False, capture_statistical: bool = True, capture_indices: bool = True):
        self.keep_values = keep_values
        self.capture_statistical = capture_statistical
        self.capture_indices = capture_indices

    def fit(self, X, y):
        """Not used.

        """

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Extract statistical information of samples.

        Parameters
        ----------
        X : ArrayLike
            The sample used to extract the information, with shape (n_samples, n_features, ).

        Returns
        -------
        ArrayLike
            An array with the statistical information about the samples. If 
            `keep_values` parameter is set, the statistical information will be
            concatenated along the input sample.

        """

        datas = []

        for data in X:
            values = []
            if self.capture_statistical:
                values += [
                    np.mean(data),
                    np.std(data),
                    np.mean(np.absolute(data - np.mean(data))),
                    np.min(data),
                    np.max(data),
                    np.max(data)-np.min(data),
                    np.median(data),
                    np.median(np.absolute(data - np.median(data))),
                    np.percentile(data, 75) - np.percentile(data, 25),
                    np.sum(data < 0),
                    np.sum(data > 0),
                    np.sum(data > np.mean(data)),
                    len(find_peaks(data)[0]),
                    stats.skew(data),
                    stats.kurtosis(data),
                    np.sum(data**2)/100,
                ]
            if self.capture_indices:
                values += [
                    np.argmax(data),
                    np.argmin(data),
                    np.argmax(data)-np.argmin(data)
                ]

            values = np.array(values)
            if self.keep_values:
                values = np.concatenate([data, values])
            datas.append(values)

        return np.array(datas)

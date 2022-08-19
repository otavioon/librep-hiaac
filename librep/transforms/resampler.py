import numpy as np
from scipy import signal

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike


class SimpleResampler(Transform):
    """Resample a single sample using `scipy.signal.resample` method.

    Parameters
    ----------
    new_sample_size : int
        The new number of points.

    """

    def __init__(self, new_sample_size: int):
        self.new_sample_size = new_sample_size

    def fit(self, X: ArrayLike, y: ArrayLike = None):
        """Not used.

        """

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Resample a single signal sample.

        Parameters
        ----------
        X : ArrayLike
            The signal sample with shape: (n_features, )

        Returns
        -------
        ArrayLike
            The resampled sample with shape: (new_sample_size, ).

        """
        
        return signal.resample(X, self.new_sample_size)

import numpy as np

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike


class AutoCorrelation(Transform):
    """Calculates the autocorrelation of a sample.

    Parameters
    ----------
    mode : str
        Mode of transform. See `np.correlate`.

    """

    def __init__(self, mode: str = "full"):
        self.mode = mode

    def fit(self, X: ArrayLike, y: ArrayLike = None):
        """Not used.

        """
        pass

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Returns the autocorrelation of a sample.

        Parameters
        ----------
        X : ArrayLike
            A single sample to be transfomed with shape (n_features, ).

        Returns
        -------
        ArrayLike
            The transformed sample.

        """

        data = np.correlate(X, X, mode=self.mode)
        data = data[data.size // 2:]
        return data

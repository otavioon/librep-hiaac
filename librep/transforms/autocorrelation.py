import numpy as np

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike


class AutoCorrelation(Transform):
    def __init__(self, mode: str = "full"):
        self.mode = mode

    def fit(self, X: ArrayLike, y: ArrayLike = None):
        pass

    def transform(self, X: ArrayLike):
        data = np.correlate(X, X, mode=self.mode)
        data = data[data.size // 2:]
        return data
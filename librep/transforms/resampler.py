import numpy as np
from scipy import signal

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike


class SimpleResampler(Transform):
    def __init__(self, new_sample_size: int):
        self.new_sample_size = new_sample_size

    def fit(self, X, y):
        pass

    def transform(self, X: ArrayLike) -> ArrayLike:
        return signal.resample(X, self.new_sample_size)
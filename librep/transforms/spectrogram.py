import numpy as np
from scipy import signal

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike


class Spectrogram(Transform):

    def __init__(self,
                 fs: float = 100,
                 segment_size: int = 50,
                 overlap: int = 30):
        self.fs = fs
        self.segment_size = segment_size
        self.overlap = overlap

    # TODO
    def transform(self, X: ArrayLike):
        Sxx, f, t = signal.spectrogram(X,
                                       fs=self.fs,
                                       nperseg=self.segment_size,
                                       noverlap=self.overlap)
        return (Sxx, f, t)
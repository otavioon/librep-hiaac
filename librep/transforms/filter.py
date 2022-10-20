import numpy as np
from scipy import signal
from scipy.signal import butter, sosfilt

from librep.config.type_definitions import ArrayLike
from librep.base.transform import Transform


class ButterWorthFilter(Transform):
    """

    Parameters
    ----------

    Examples
    ----------


    """
    def __init__(self, 
               N: int = 3, 
               Wn: float = 0.3, 
               btype: str = 'high', 
               fs: float = 20, 
               output: str = 'sos'):
        self.N = N
        self.Wn = Wn
        self.btype = btype
        self.fs = fs
        self.output = output

    def transform(self, X: ArrayLike) -> ArrayLike:

        sos = signal.butter(self.N, self.Wn, btype=self.btype, fs=self.fs, output=self.output)
        datas = []
        for sample in X:
            filtered = signal.sosfilt(sos, sample)
            datas.append(filtered)
        datas = np.array(datas)

        return datas

        # raise NotImplementedError


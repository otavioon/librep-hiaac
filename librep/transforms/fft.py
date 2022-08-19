import numpy as np
from scipy import fftpack

from librep.base.transform import InvertibleTransform
from librep.config.type_definitions import ArrayLike


class FFT(InvertibleTransform):
    """Performs the DFT at one sample.

    Parameters
    ----------
    transpose : bool
        Description of parameter `transpose` (the default is False).
    absolute : bool
        Description of parameter `absolute` (the default is True).
    centered : bool
        Description of parameter `centered` (the default is False).

    Attributes
    ----------
    transpose
    absolute
    centered

    Example
    --------
    >>> FFT()

    """

    def __init__(self,
                 transpose: bool = False,
                 absolute: bool = True,
                 centered: bool = False):
        # Transpose == False:
        # - AccFreqDomEmbedding
        # Transpose == True:
        # - AccFreqDomMultiChannel
        # - AccGyrFreqDomMultiChannel
        self.transpose = transpose
        self.absolute = absolute
        self.centered = centered

    def fit(self, X: ArrayLike, y: ArrayLike = None):
        pass

    def transform(self, X: ArrayLike):
        """Transform a sample using FFT.

        Parameters
        ----------
        X : ArrayLike
            Description of parameter `X`.

        Returns
        -------
        type
            Description of returned object.

        Raises
        ------
        ExceptionName
            Why the exception is raised.

        """
        data = fftpack.fft(X)
        if self.absolute:
            data = np.abs(data)
        if self.centered:
            data = data[:len(data)//2]

        return data.T if self.transpose else data

    def inverse_transform(self, X: ArrayLike):
        data = X.T if self.transpose else X
        # TODO absolute?
        return fftpack.ifft(data)

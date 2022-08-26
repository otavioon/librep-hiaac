import numpy as np
from scipy import fftpack

from librep.base.transform import InvertibleTransform
from librep.config.type_definitions import ArrayLike


class FFT(InvertibleTransform):
    """Performs the DFT at one sample.

    Parameters
    ----------
    transpose : bool
        Return the transpose of the transformed sample (the default is False).
    absolute : bool
        Return the absolute values instead of a complex number (the default is True).
    centered : bool
        If True, returns the only the first half of the transformed sample, as
        FFT is symmetric (the default is False).

    Examples
    ----------
    >>> time_sample = np.arange(256)
    >>> fft_transform = FFT(centered=True)
    >>> fft_sample = fft_transform.fit_transform(time_sample)
    >>> fft_sample.shape
    (128, )

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
        """Not used.

        """

        return X

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transform a sample using FFT.

        Parameters
        ----------
        X : ArrayLike
            The samples to be transfomed with shape (n_features, n_features, ).

        Returns
        -------
        ArrayLike
            The transformed sample.

        """

        datas = []
        for data in X:
            data = fftpack.fft(data)
            if self.absolute:
                data = np.abs(data)
            if self.centered:
                data = data[:len(data)//2]
            datas.append(data)

        data = np.array(datas)
        return data.T if self.transpose else data

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        """Transform a sample with the inverse FFT.

        Parameters
        ----------
        X : ArrayLike
            The samples to be transfomed with shape (n_features, ).

        Returns
        -------
        ArrayLike
            The transformed sample.

        """

        raise NotImplementedError
#         data = X.T if self.transpose else X
#         # TODO absolute?
#         return fftpack.ifft(data)

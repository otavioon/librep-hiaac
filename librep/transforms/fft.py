import numpy as np
from scipy import fftpack

from librep.base.transform import InvertibleTransform
from librep.config.type_definitions import ArrayLike


class FFT(InvertibleTransform):

    def __init__(self, transpose: bool = False, absolute: bool = True):
        # Transpose == False:
        # - AccFreqDomEmbedding
        # Transpose == True:
        # - AccFreqDomMultiChannel
        # - AccGyrFreqDomMultiChannel
        self.transpose = transpose
        self.absolute = absolute

    def fit(self, X: ArrayLike, y: ArrayLike = None):
        pass

    def transform(self, X: ArrayLike):
        data = fftpack.fft(X)
        if self.absolute:
            data = np.abs(data)

        return data.T if self.transpose else data

    def inverse_transform(self, X: ArrayLike):
        data = X.T if self.transpose else X
        # TODO absolute?
        return fftpack.ifft(data)

    
# class TransformNode:
#     def __init__(self, transform, action: str = "fit_transform"):
#         self.transform = transform
#         self.action = action
        
#     def run(self, *args, **kwargs):
#         if action == "fit_transform":
#             return self.fit_transform(*args, **kwargs)
        
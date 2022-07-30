# mobile_sensors_dataset_base.py
import numpy as np
from ..embeddings.basic_emb_dataset_types import *

from scipy import fftpack

class AccFreqDomSumEmbedding_DataSubSet(Embedding_DataSubSet): 
    def __init__(self, dss: Canonical_DataSubSet):
        """Initialize the data subset converting from the canonical data subset"""
        self.samples = [self.convert_sample(i) for i in dss.samples]
        self.labels = dss.labels
        self.embedding_name = "AccFreqDomSumEmbedding"
        
    def convert_time_series(self, ts):
        X = fftpack.fft(ts.data)
        #freqs = fftpack.fftfreq(len(acc_norm)) * ts.samp_freq
        return np.abs(X)
        
    def convert_sample(self,s: Canonical_Sample):
        """For each acc norm time series, convert it to its frequency domain"""
        return self.convert_time_series(s.acc.x) + self.convert_time_series(s.acc.y) + self.convert_time_series(s.acc.z)

class AccFreqDomSumEmbedding_DataSet(Embedding_DataSet): 
    def __init__(self):
        """Initialize the dataset converting from the canonical dataset"""
        self.embedding_name = "AccFreqDomSumEmbedding"
        self.train      = None
        self.validation = None
        self.test       = None

    def import_from_canonical(self, ds: Canonical_DataSet): 
        self.train      = AccFreqDomSumEmbedding_DataSubSet(ds.train)
        self.validation = AccFreqDomSumEmbedding_DataSubSet(ds.validation)
        self.test       = AccFreqDomSumEmbedding_DataSubSet(ds.test)


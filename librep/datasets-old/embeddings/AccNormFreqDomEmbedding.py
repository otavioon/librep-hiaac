# mobile_sensors_dataset_base.py
import numpy as np
from ..embeddings.basic_emb_dataset_types import *

from scipy import fftpack

class AccNormFreqDomEmbedding_DataSubSet(Embedding_DataSubSet): 
    def __init__(self, dss: Canonical_DataSubSet):
        """Initialize the data subset converting from the canonical data subset"""
        self.samples = [self.convert_sample(i) for i in dss.samples]
        self.labels = dss.labels
        self.embedding_name = "AccNormFreqDomEmbedding"

    def convert_time_series(self, ts):
        X = fftpack.fft(ts.data)
        return np.abs(X)
        
    def convert_sample(self,s: Canonical_Sample):
        """For each acc norm time series, convert it to its frequency domain"""
        return self.convert_time_series(s.acc.norm())
    
class AccNormFreqDomEmbedding_DataSet(Embedding_DataSet): 
    def __init__(self):
        """Initialize the dataset converting from the canonical dataset"""
        self.embedding_name = "AccNormFreqDomEmbedding"
        self.train      = None
        self.validation = None
        self.test       = None

    def import_from_canonical(self, ds: Canonical_DataSet):
        self.train      = AccNormFreqDomEmbedding_DataSubSet(ds.train)
        self.validation = AccNormFreqDomEmbedding_DataSubSet(ds.validation)
        self.test       = AccNormFreqDomEmbedding_DataSubSet(ds.test)



# mobile_sensors_dataset_base.py
import numpy as np
from ..embeddings.basic_emb_dataset_types import *

from scipy import fftpack

#####################################################################################################
# AccFreqDomMultiChannelEmbedding: Acc represented in the frequency domain in the multichannel format
#####################################################################################################
class AccFreqDomMultiChannelEmbedding_DataSubSet(Embedding_DataSubSet): 
    def __init__(self, dss: Canonical_DataSubSet):
        """Initialize the data subset converting from the canonical data subset"""
        self.samples = [self.convert_sample(i) for i in dss.samples]
        self.labels = dss.labels
        self.embedding_name = "AccFreqDomMultiChannelEmbedding"
        
    def convert_time_series(self, ts):
        X = fftpack.fft(ts.data)
        #freqs = fftpack.fftfreq(len(acc_norm)) * ts.samp_freq
        return np.abs(X)
        
    def convert_sample(self,s: Canonical_Sample):
        """For each acc norm time series, convert it to its frequency domain and save to multichannel format"""
        return np.array((self.convert_time_series(s.acc.x), 
                         self.convert_time_series(s.acc.y), 
                         self.convert_time_series(s.acc.z))).T

class AccFreqDomMultiChannelEmbedding_DataSet(Embedding_DataSet): 
    def __init__(self):
        """Initialize the dataset converting from the canonical dataset"""
        self.embedding_name = "AccFreqDomMultiChannelEmbedding"
        self.train      = None
        self.validation = None
        self.test       = None

    def import_from_canonical(self, ds: Canonical_DataSet):
        self.train      = AccFreqDomMultiChannelEmbedding_DataSubSet(ds.train)
        self.validation = AccFreqDomMultiChannelEmbedding_DataSubSet(ds.validation)
        self.test       = AccFreqDomMultiChannelEmbedding_DataSubSet(ds.test)


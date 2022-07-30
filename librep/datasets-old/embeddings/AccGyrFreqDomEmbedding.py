# mobile_sensors_dataset_base.py
import numpy as np
from ..embeddings.basic_emb_dataset_types import *

from scipy import fftpack

##########################################################################################
# AccGyrFreqEmbedding: Acc and Gyr represented in the frequency domain
# AccNormFreqEmbedding: Acc Norm represented in the frequency domain
##########################################################################################
class AccGyrFreqDomEmbedding_DataSubSet(Embedding_DataSubSet): 
    def __init__(self, dss: Canonical_DataSubSet):
        """Initialize the data subset converting from the canonical data subset"""
        self.samples = [self.convert_sample(i) for i in dss.samples]
        self.labels = dss.labels
        self.embedding_name = "AccGyrFreqDomEmbedding"
        
    def convert_time_series(self, ts):
        X = fftpack.fft(ts.data)
        #freqs = fftpack.fftfreq(len(acc_and gyr)) * ts.samp_freq
        return np.abs(X)
        
    def convert_sample(self,s: Canonical_Sample):
        """For each acc and gyr time series, convert it to its frequency domain"""
        #freqs = fftpack.fftfreq(len(acc_norm)) * ts.samp_freq
        return np.abs(X)
        
    def convert_sample(self,s: Canonical_Sample):
        """For each acc norm time series, convert it to its frequency domain"""
        return np.concatenate((self.convert_time_series(s.acc.x), 
                               self.convert_time_series(s.acc.y), 
                               self.convert_time_series(s.acc.z), 
                               self.convert_time_series(s.gyr.x), 
                               self.convert_time_series(s.gyr.y), 
                               self.convert_time_series(s.gyr.z)))

class AccGyrFreqDomEmbedding_DataSet(Embedding_DataSet): 
    def __init__(self):
        """Initialize the dataset converting from the canonical dataset"""
        self.embedding_name = "AccGyrFreqDomEmbedding"
        self.train      = None
        self.validation = None
        self.test       = None

    def import_from_canonical(self, ds: Canonical_DataSet):
        self.train      = AccGyrFreqDomEmbedding_DataSubSet(ds.train)
        self.validation = AccGyrFreqDomEmbedding_DataSubSet(ds.validation)
        self.test       = AccGyrFreqDomEmbedding_DataSubSet(ds.test)


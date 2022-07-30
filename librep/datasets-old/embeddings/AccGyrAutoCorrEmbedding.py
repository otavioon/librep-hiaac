# mobile_sensors_dataset_base.py
import numpy as np
from ..embeddings.basic_emb_dataset_types import *


##########################################################################################
# AccGyrAutoCorrEmbedding: Acc and Gyr represented in the autocorrelation domain
##########################################################################################
class AccGyrAutoCorrEmbedding_DataSubSet(Embedding_DataSubSet): 
    def __init__(self, dss: Canonical_DataSubSet):
        """Initialize the data subset converting from the canonical data subset"""
        self.samples = [self.convert_sample(i) for i in dss.samples]
        self.labels = dss.labels
        self.embedding_name = "AccGyrAutoCorrEmbedding"
        
    def calculate_autocorrelation(self, ts):
        X = np.correlate(ts.data, ts.data, mode = 'full')
        X = X[X.size // 2 :]
        return X
        
    def convert_sample(self,s: Canonical_Sample):
        """For each acc and gyr time series, convert it to its autocorrelation domain"""
        return np.concatenate((self.calculate_autocorrelation(s.acc.x), 
                               self.calculate_autocorrelation(s.acc.y), 
                               self.calculate_autocorrelation(s.acc.z), 
                               self.calculate_autocorrelation(s.gyr.x), 
                               self.calculate_autocorrelation(s.gyr.y), 
                               self.calculate_autocorrelation(s.gyr.z)))

class AccGyrAutoCorrEmbedding_DataSet(Embedding_DataSet): 
    def _init_(self):
        """Initialize the dataset converting from the canonical dataset"""
        self.embedding_name = "AccGyrAutoCorrEmbedding"
        self.train      = None
        self.validation = None
        self.test       = None

    def import_from_canonical(self, ds: Canonical_DataSet):
        self.train      = AccGyrAutoCorrEmbedding_DataSubSet(ds.train)
        self.validation = AccGyrAutoCorrEmbedding_DataSubSet(ds.validation)
        self.test       = AccGyrAutoCorrEmbedding_DataSubSet(ds.test)
        self.embedding_name = "AccGyrAutoCorrEmbedding"
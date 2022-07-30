# mobile_sensors_dataset_base.py
import numpy as np
from ..embeddings.basic_emb_dataset_types import *

##########################################################################################
# AccNormEmbedding: For each vector defined by x[i], y[i], z[i], compute its norm
##########################################################################################
class AccNormEmbedding_DataSubSet(Embedding_DataSubSet): 
    def __init__(self, dss: Canonical_DataSubSet):
        """Initialize the data subset converting from the canonical data subset"""
        self.samples = [self.convert_sample(i) for i in dss.samples]
        self.labels = dss.labels
        self.embedding_name = "AccNormEmbedding"
        
    def convert_sample(self,s: Canonical_Sample):
        """For each vector defined by x[i], y[i], z[i], compute its norm"""
        return s.acc.norm()

class AccNormEmbedding_DataSet(Embedding_DataSet): 
    def __init__(self):
        """Initialize the dataset converting from the canonical dataset"""
        self.embedding_name = "AccNormEmbedding"
        self.train      = None
        self.validation = None
        self.test       = None

    def import_from_canonical(self, ds: Canonical_DataSet):
        self.train      = AccNormEmbedding_DataSubSet(ds.train)
        self.validation = AccNormEmbedding_DataSubSet(ds.validation)
        self.test       = AccNormEmbedding_DataSubSet(ds.test)

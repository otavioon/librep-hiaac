# mobile_sensors_dataset_base.py
import numpy as np
from ..embeddings.basic_emb_dataset_types import *

##########################################################################################
# RawAccGyrEmbedding: concatenates the x, y, and y samples from the accelerometer
##########################################################################################
class RawAccGyrEmbedding_DataSubSet(Embedding_DataSubSet): 
    def __init__(self, dss: Canonical_DataSubSet):
        """Initialize the data subset converting from the canonical data subset"""
        self.samples = [self.convert_sample(i) for i in dss.samples]
        self.labels = dss.labels
        self.embedding_name = "RawAccGyrEmbedding"
        
    def convert_sample(self,s: Canonical_Sample):
        acc_x_data = s.acc.x.ravel()
        acc_y_data = s.acc.y.ravel()
        acc_z_data = s.acc.z.ravel()
        gyr_x_data = s.gyr.x.ravel()
        gyr_y_data = s.gyr.y.ravel()
        gyr_z_data = s.gyr.z.ravel()
        acc_gyr_time_embedding = []
        for i in range(len(acc_x_data)):
            acc_gyr_time_embedding.extend([acc_x_data[i], acc_y_data[i], acc_z_data[i],
                                          gyr_x_data[i], gyr_y_data[i], gyr_z_data[i]])
        return np.array(acc_gyr_time_embedding)
        """Concatenates the x, y, and y accelerometer samples"""
        return np.concatenate((s.acc.x.ravel(),s.acc.y.ravel(),s.acc.z.ravel(),
                               s.gyr.x.ravel(),s.gyr.y.ravel(),s.gyr.z.ravel()))

class RawAccGyrEmbedding_DataSet(Embedding_DataSet): 
    def __init__(self):
        """Initialize the dataset converting from the canonical dataset"""
        self.embedding_name = "RawAccGyrEmbedding"
        self.train      = None
        self.validation = None
        self.test       = None

    def import_from_canonical(self, ds: Canonical_DataSet):
        self.train      = RawAccGyrEmbedding_DataSubSet(ds.train)
        self.validation = RawAccGyrEmbedding_DataSubSet(ds.validation)
        self.test       = RawAccGyrEmbedding_DataSubSet(ds.test)

# mobile_sensors_dataset_base.py
import numpy as np
import pandas as pd
from librep.datasets.basic_dataset_types import Label
from librep.datasets.embeddings.basic_emb_dataset_types import *
from librep.datasets.embeddings.RawAccGyrEmbedding import RawAccGyrEmbedding_DataSet
import umap

class UMAPEmbeddingU_DataSubSet(Embedding_DataSubSet): 
    def __init__(self, samples: pd.DataFrame, labels: [Label], emb_name):
        """Initialize the data subset converting from the canonical data subset"""
        self.embedding_name = emb_name
        if len(labels) != len(samples):
            raise NameError("Data sub set must have the same number of samples ({}) " + \
                            "and labels ({})".format(len(self.labels,self.samples)))
        self.labels = labels
        self.samples = samples.tolist()

    def get_X(self): 
        """ Return a list of saples as flattened arrays. """
        return self.samples

    def display_sample(self, samples_indices : [], label_to_str=None, sharey = True):
        """Generic method to display samples"""
        nsamples = len(samples_indices)
        for i in range(nsamples):
            sample_idx = samples_indices[i]
            if label_to_str:
                l = label_to_str(self.labels[sample_idx])
            else:
                l = str(self.labels[sample_idx])
            print("Sample {} - Data:".format(sample_idx), self.samples[sample_idx], "- Label:", l)
    
class UMAPEmbeddingU_DataSet(Embedding_DataSet): 
    def __init__(self, flattenable_ds_class = RawAccGyrEmbedding_DataSet, n_components = 2):
        """Initialize the dataset converting from the canonical dataset"""
        self.train      = None
        self.validation = None
        self.test       = None
        self.flattenable_ds_class = flattenable_ds_class
        self.n_components = n_components
        self.embedding_name = "UMAPEmbeddingU-{}comp-empty".format(self.n_components)
        
    def import_from_canonical(self, dss: Canonical_DataSet):
        # Convert the canonical dataset into a flattenable ds so we can train the umap predictor
        ft_ds = self.flattenable_ds_class()
        self.embedding_name = "UMAPEmbeddingU-{}comp-{}".format(self.n_components,ft_ds.embedding_name)
        ft_ds.import_from_canonical(dss)
        
        # Train the umap predictor using the train and validation data subsets
        self.umap_model = umap.UMAP(n_components = self.n_components) # umap.UMAP(n_neighbors=5, random_state=42)
        X = np.concatenate((ft_ds.train.get_X(),ft_ds.validation.get_X()))
        y = np.concatenate((ft_ds.train.get_y(),ft_ds.validation.get_y()))
        trans =  self.umap_model.fit(X)
        
        # Apply the umap predictor to the dataset
        self.test       = UMAPEmbeddingU_DataSubSet(trans.transform(ft_ds.test.get_X()), 
                                                   ft_ds.test.get_y(),self.embedding_name)
        self.validation = UMAPEmbeddingU_DataSubSet(trans.transform(ft_ds.validation.get_X()), 
                                                   ft_ds.validation.get_y(),self.embedding_name)
        self.train      = UMAPEmbeddingU_DataSubSet(trans.transform(ft_ds.train.get_X()), 
                                                   ft_ds.train.get_y(),self.embedding_name)

import numpy as np


class Dataset:
    def __getitem__(self, index: int) -> tuple:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class FlattenableDataset(Dataset):
    def flatten(self) -> tuple:
        pass



# class DataSubSet:
#     """A data subset contains a set of samples and a set of corresponding labels
#     """
#     def __init__(self, samples : [Sample] = None, labels : [Label] = None):
#         if not samples: samples = []
#         if not labels: labels = []
#         self.samples = samples
#         self.labels  = labels
#         if len(self.samples) != len(self.labels):
#             raise NameError("A canonical DataSubSet must have the same amounts of samples and labels. len(samples) = {}, len(labels) = {}.".format(len(samples), len(labels)))

#     def append_sample(self, sample : Sample, label : Label):
#         """Appends a new sample and its respective label to the data subset"""
#         self.samples.append(sample)
#         self.labels.append(label)

class DataSet:
    """A dataset contains three data subsets: train, validation, and test"""
    def __init__(self, train : DataSubSet = None, validation : DataSubSet = None, test : DataSubSet = None):
        if not train: train = DataSubSet()
        if not validation: validation = DataSubSet()
        if not test: test = DataSubSet()
        self.train      = train
        self.validation = validation
        self.test       = test

    def write_to_dir(self, dirname : str, save_to_csv : bool = False):
        """ Store the data subsets train, validation, and 
            test into files at a given directory. """
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname,"train.pkl"), "wb") as fh: pickle.dump(self.train, fh)
        with open(os.path.join(dirname,"validation.pkl"), "wb") as fh: pickle.dump(self.validation, fh)
        with open(os.path.join(dirname,"test.pkl"), "wb") as fh: pickle.dump(self.test, fh)

    def read_from_dir(self, dirname : str): 
        """ Read the data subsets train, validation, and 
            test from files at a given directory. """
        with open(os.path.join(dirname,"train.pkl"), "rb") as fh: self.train = pickle.load(fh)
        with open(os.path.join(dirname,"validation.pkl"), "rb") as fh: self.validation = pickle.load(fh)
        with open(os.path.join(dirname,"test.pkl"), "rb") as fh: self.test = pickle.load(fh)

##########################################################################################
# Base classes for Flattenable datasets
##########################################################################################
import os
import pickle

class Flattenable_Sample:
    """ A Flattenable Sample is a sample that can be flattened into an 1D array. """    
    def ravel(self): pass
    """ Return the sample as a contiguous flattened array, i.e., 1D array. """

class NumPy_NDArray_Sample(np.ndarray,Flattenable_Sample): pass

class Flattenable_DataSubSet(DataSubSet): 
    """ A Flattenable data subset is a data subset in which 
        the samples can be flattened into 1D arrays. """

    def __init__(self, samples : [Flattenable_Sample] = None, labels : [Label] = None):
        DataSubSet.__init__(self, samples, labels)
        
    def get_X(self): 
        """ Return a list of samples as flattened (1D arrays) arrays. """
        return [ s.ravel()  for s in self.samples]

    def get_y(self): 
        """ Return the list of labels. """
        return self.labels

class Flattenable_DataSet(DataSet): 
    """A flattenable dataset contains three flattenable 
       data subsets: train, validation, and test"""
    def __init__(self, train : Flattenable_DataSubSet = None, 
                 validation : Flattenable_DataSubSet = None, 
                 test : Flattenable_DataSubSet = None):
        if not train: train = Flattenable_DataSubSet()
        if not validation: validation = Flattenable_DataSubSet()
        if not test: test = Flattenable_DataSubSet()
        DataSet.__init__(self, train=train, validation=validation, test=test)


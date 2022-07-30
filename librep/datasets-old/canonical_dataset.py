# mobile_sensors_dataset_base.py
import numpy as np
import os
from typing import List
from librep.datasets.basic_dataset_types import Label, Sample, DataSubSet, DataSet

import matplotlib.pyplot as plt # Display sample

##########################################################################################
# Canonical dataset classes
##########################################################################################
    
class FloatTimeSeries(): 
    """A class to represent time series with float numbers"""
    def __init__(self,data : np.ndarray = None, samp_freq : np.int32 = None):
        self.data = data
        self.samp_freq = samp_freq
        if not self.samp_freq:
            print("Warning: Creating a time series without sampling information")
    def ravel(self): return self.data.ravel()

class XYZ_TimeSeries_Sample(Sample):
    def __init__(self, x: FloatTimeSeries, y: FloatTimeSeries, z: FloatTimeSeries):
        self.x = x
        self.y = y
        self.z = z
    def ravel(self): 
        return np.concatenate((self.x.ravel(),self.y.ravel(),self.z.ravel()))
    def norm(self): 
        if self.x.samp_freq != self.y.samp_freq or self.y.samp_freq != self.z.samp_freq:
            err_msg = "x, y, and z sampling frequencies must be the same: x = {}, y = {}, " + \
                      "z = {}".format(self.x.samp_freq, self.y.samp_freq, self.z.samp_freq)
            raise NameError(err_msg)
        return np.sqrt(self.x.data * self.x.data + self.y.data * self.y.data + self.z.data * self.z.data)

class ACC_Sample(XYZ_TimeSeries_Sample): pass

class GYR_Sample(XYZ_TimeSeries_Sample): pass

class Canonical_Sample(Sample):
    def __init__(self, acc : ACC_Sample = None, 
                 gyr : GYR_Sample = None):
        self.acc = acc
        self.gyr = gyr
                
class Canonical_DataSubSet(DataSubSet): 
    def __init__(self, samples : List[Canonical_Sample] = None, 
                 labels : List[Label] = None):
        DataSubSet.__init__(self, samples, labels)

    def write_to_file(self, basename : str): 
        """ Store the dataset samples and labels on files. """
        np.save(basename+".samples",self.samples)
        np.save(basename+".labels",self.labels)

    def read_from_file(self, basename : str): 
        """ Read the dataset samples and labels from files. """
        self.samples = np.load(basename+".samples.npy", allow_pickle=True)
        self.labels  = np.load(basename+".labels.npy", allow_pickle=True)

    def display_sample(self, display_samples : List[Canonical_Sample], label_to_str=None, sharey = True):
        if type(display_samples) != list:
            display_samples = [display_samples]
        nsamples = len(display_samples)
        fig, axs = plt.subplots(nrows=2, ncols=nsamples, figsize=(5*nsamples,5), 
                                sharex=True, sharey=sharey) # , sharey=True)
        def plot_XYZ_chart(ax,xyz_data,label_prefix,title):
            ax.set_title(title)
            ax.plot(xyz_data.x.data, color='r', label=label_prefix+".x")
            ax.plot(xyz_data.y.data, color='b', label=label_prefix+".y")
            ax.plot(xyz_data.z.data, color='g', label=label_prefix+".z")
            ax.legend()            
        for i in range(nsamples):
            sample_idx = display_samples[i]
            s = self.samples[sample_idx]
            l = self.labels[sample_idx]            
            if label_to_str:
                sample_title = "Sample {} : Label {}".format(sample_idx, label_to_str(l))
            else:
                sample_title = "Sample {} : Label {}".format(sample_idx, l)
            if nsamples == 1:
                plot_XYZ_chart(axs[0], s.acc, "acc", sample_title+"\nAccelerometer")
                plot_XYZ_chart(axs[1], s.gyr, "gyr", "Gyroscope")
            else:
                plot_XYZ_chart(axs[0][i], s.acc, "acc", sample_title+"\nAccelerometer")
                plot_XYZ_chart(axs[1][i], s.gyr, "gyr", "Gyroscope")
        plt.show()
        
class Canonical_DataSet(DataSet): 
    def __init__(self, train : Canonical_DataSubSet = None, 
                 validation : Canonical_DataSubSet = None, 
                 test : Canonical_DataSubSet = None):
        if not train: train = Canonical_DataSubSet()
        if not validation: validation = Canonical_DataSubSet()
        if not test: test = Canonical_DataSubSet()
        DataSet.__init__(self, train=train, validation=validation, test=test)

from librep.datasets.basic_dataset_types import Flattenable_DataSet, Flattenable_DataSubSet
from librep.datasets.canonical_dataset import Canonical_DataSet, Canonical_DataSubSet, Canonical_Sample
     
import matplotlib.pyplot as plt # Display sample

##########################################################################################
# Base Embedding dataset abstract classes
# - All embedding classes must derive from these classes and implement their 
#   respective constructors to convert from canonical data sets.
# - Sample types must derive from Flattenable Samples.
##########################################################################################

class Embedding_DataSubSet(Flattenable_DataSubSet):
    def __init__(self, dss: Canonical_DataSubSet):
        raise NameError("This method must be implemented by the child class")
    
    def display_sample(self, samples_indices : [], label_to_str=None, sharey = True):
        """Generic method to display samples"""
        nsamples = len(samples_indices)
        fig, axs = plt.subplots(nrows=1, ncols=nsamples, figsize=(5*nsamples,5), sharey=sharey)
        def plot_sample(ax,s,l):
            if label_to_str:
                sample_title = "Sample {} : Label {}".format(sample_idx, label_to_str(l))
            else:
                sample_title = "Sample {} : Label {}".format(sample_idx, l)
            ax.set_title(sample_title)
            ax.plot(s, color='C0')
            #axs.set_xlabel("")
            #axs.set_ylabel("")

        for i in range(nsamples):
            sample_idx = samples_indices[i]
            if nsamples == 1:
                plot_sample(axs,self.samples[sample_idx],self.labels[sample_idx])
            else:
                plot_sample(axs[i],self.samples[sample_idx],self.labels[sample_idx])
        plt.show()

class Embedding_DataSet(Flattenable_DataSet):

    def import_from_canonical(self, ds: Canonical_DataSet): pass

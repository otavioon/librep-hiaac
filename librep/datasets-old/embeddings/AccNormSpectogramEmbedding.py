import numpy as np
from ..embeddings.basic_emb_dataset_types import *
from scipy import signal
from IPython.display import display

class AccNormSpectogramEmbedding_DataSubSet(Embedding_DataSubSet): 
    def __init__(self, dss: Canonical_DataSubSet, nperseg = 50, noverlap = 30):
        """Initialize the data subset converting from the canonical data subset"""
        self.embedding_name = "AccNormSpectogramEmbedding-{}-{}".format(nperseg,noverlap)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.samples = [self.convert_sample(i) for i in dss.samples]
        self.labels = dss.labels

    def display_sample(self, display_samples : [], label_to_str=None, sharey = True):
        nsamples = len(display_samples)
        fig, axs = plt.subplots(nrows=2, ncols=nsamples, figsize=(5*nsamples,5), sharey=sharey) # , sharey=True)
        for i in range(nsamples):
            sample_idx = display_samples[i]
            s = self.samples[sample_idx]
            l = self.labels[sample_idx]
            if label_to_str:
                sample_title = "Sample {} : Label {}".format(sample_idx, label_to_str(l))
            else:
                sample_title = "Sample {} : Label {}".format(sample_idx, l)

            def plot_charts(ax_0, ax_1, s, sample_title):
                ax_0.set_title(sample_title)
                ax_0.pcolormesh(s[2], s[1], s[0], shading='gouraud')
                ax_0.set_ylabel('Frequency [Hz]')
                ax_1.set_xlabel('Time [sec]')
                ax_1.set_ylabel('Frequency [Hz]')
                ax_1.imshow(s[0], cmap='hot', interpolation='nearest')

            if nsamples == 1: plot_charts(axs[0], axs[1], s, sample_title)
            else:             plot_charts(axs[0][i], axs[1][i], s, sample_title)

        plt.show()        
        
    def gen_spectogram(self, s : Canonical_Sample):
        """TODO: Describe"""
        #_, _, Sxx0 = signal.spectrogram(s.acc.x.data, 100, nperseg=64, noverlap=32)
        #_, _, Sxx1 = signal.spectrogram(s.acc.y.data, 100, nperseg=64, noverlap=32)
        #f, t, Sxx2 = signal.spectrogram(s.acc.z.data, 100, nperseg=64, noverlap=32)
        #return (Sxx0+Sxx1+Sxx2, f, t)
        
        #f, t, Sxx = signal.spectrogram(s.acc.x.data+s.acc.y.data+s.acc.z.data, 100, nperseg=64, noverlap=32)
        
        f, t, Sxx = signal.spectrogram(s.acc.norm(), 100, nperseg=self.nperseg, noverlap=self.noverlap)
        return (Sxx, f, t)
    
    def convert_sample(self,s: Canonical_Sample) : return self.gen_spectogram(s)

    def get_X(self): 
        """ Return a list of saples as flattened arrays. """
        # s[0] == Sxx
        #array.reshape(-1, 1)
        #array.ravel()
        return [ s[0].ravel()  for s in self.samples]

class AccNormSpectogramEmbedding_DataSet(Embedding_DataSet): 
    def __init__(self, nperseg = 50, noverlap = 30):
        """Initialize the dataset converting from the canonical dataset"""
        self.embedding_name = "AccNormSpectogramEmbedding-{}-{}".format(nperseg,noverlap)
        self.train      = None
        self.validation = None
        self.test       = None
        self.nperseg    = nperseg
        self.noverlap   = noverlap

    def import_from_canonical(self, ds: Canonical_DataSet):
        self.train      = AccNormSpectogramEmbedding_DataSubSet(ds.train, nperseg=self.nperseg, 
                                                                noverlap=self.noverlap)
        self.validation = AccNormSpectogramEmbedding_DataSubSet(ds.validation, nperseg=self.nperseg, 
                                                                noverlap=self.noverlap)
        self.test       = AccNormSpectogramEmbedding_DataSubSet(ds.test, nperseg=self.nperseg, 
                                                                noverlap=self.noverlap)

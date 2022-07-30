import numpy as np
from .basic_emb_dataset_types import *
from scipy import signal
from IPython.display import display

class AccSpectrogramEmbedding_DataSubSet(Embedding_DataSubSet): 
    def __init__(self, dss: Canonical_DataSubSet, nperseg = 50, noverlap = 30):
        """Initialize the data subset converting from the canonical data subset"""
        self.embedding_name = "AccSpectrogramEmbedding-{}-{}".format(nperseg,noverlap)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.samples = [self.convert_sample(i) for i in dss.samples]
        self.labels = dss.labels

    def display_sample(self, display_samples:list = [], label_to_str=None, sharey = True):
        nsamples = len(display_samples)
        fig, axs = plt.subplots(nrows=2, ncols=3*nsamples, figsize=(5*3*nsamples,5), sharey=sharey) # , sharey=True)
        for i in range(nsamples):
            sample_idx = display_samples[i]
            s = self.samples[sample_idx]
            l = self.labels[sample_idx]
            if label_to_str:
                sample_title = "Sample {} : Label {}".format(sample_idx, label_to_str(l))
            else:
                sample_title = "Sample {} : Label {}".format(sample_idx, l)

            def plot_charts(ax_0, ax_1, ax_2, ax_3, ax_4, ax_5, s, sample_title):
                ax_0.set_title(sample_title)
                ax_0.pcolormesh(s[2], s[1], s[0][0], shading='gouraud')
                ax_0.set_ylabel('Frequency [Hz]')
                ax_1.set_xlabel('Time [sec]')
                ax_1.set_ylabel('Frequency [Hz]')
                ax_1.imshow(s[0][0], cmap='hot', interpolation='nearest')

                ax_2.set_title(sample_title)
                ax_2.pcolormesh(s[2], s[1], s[0][1], shading='gouraud')
                ax_2.set_ylabel('Frequency [Hz]')
                ax_3.set_xlabel('Time [sec]')
                ax_3.set_ylabel('Frequency [Hz]')
                ax_3.imshow(s[0][1], cmap='hot', interpolation='nearest')

                ax_4.set_title(sample_title)
                ax_4.pcolormesh(s[2], s[1], s[0][2], shading='gouraud')
                ax_4.set_ylabel('Frequency [Hz]')
                ax_5.set_xlabel('Time [sec]')
                ax_5.set_ylabel('Frequency [Hz]')
                ax_5.imshow(s[0][2], cmap='hot', interpolation='nearest')

            if nsamples == 1: plot_charts(axs[0][0], axs[1][0], axs[0][1], axs[1][1], axs[0][2], axs[1][2], s, sample_title)
            else:             plot_charts(axs[0][0+3*i], axs[1][0+3*i], axs[0][1+3*i], axs[1][1+3*i], axs[0][2+3*i], axs[1][2+3*i], s, sample_title)

        plt.show()        
        
    def gen_spectrogram(self, s : Canonical_Sample):
        """TODO: Describe"""
        _, _, Sxx_x = signal.spectrogram(s.acc.x.data, 100, nperseg=self.nperseg, noverlap=self.noverlap)
        _, _, Sxx_y = signal.spectrogram(s.acc.y.data, 100, nperseg=self.nperseg, noverlap=self.noverlap)
        f, t, Sxx_z = signal.spectrogram(s.acc.z.data, 100, nperseg=self.nperseg, noverlap=self.noverlap)
        return ((Sxx_x,Sxx_y,Sxx_z), f, t)
    
    def convert_sample(self,s: Canonical_Sample) : return self.gen_spectrogram(s)

    def get_X(self): 
        return [np.hstack((s[0][0].ravel(), s[0][1].ravel(), s[0][2].ravel())) for s in self.samples]

class AccSpectrogramEmbedding_DataSet(Embedding_DataSet): 
    def __init__(self, nperseg = 50, noverlap = 30):
        """Initialize the dataset converting from the canonical dataset"""
        self.embedding_name = "AccSpectrogramEmbedding-{}-{}".format(nperseg,noverlap)
        self.train      = None
        self.validation = None
        self.test       = None
        self.nperseg    = nperseg
        self.noverlap   = noverlap

    def import_from_canonical(self, ds: Canonical_DataSet):
        self.train      = AccSpectrogramEmbedding_DataSubSet(ds.train, nperseg=self.nperseg, 
                                                                noverlap=self.noverlap)
        self.validation = AccSpectrogramEmbedding_DataSubSet(ds.validation, nperseg=self.nperseg, 
                                                                noverlap=self.noverlap)
        self.test       = AccSpectrogramEmbedding_DataSubSet(ds.test, nperseg=self.nperseg, 
                                                                noverlap=self.noverlap)

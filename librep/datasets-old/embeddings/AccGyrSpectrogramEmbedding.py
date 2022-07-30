import numpy as np
from .basic_emb_dataset_types import *
from scipy import signal
from IPython.display import display

class AccGyrSpectrogramEmbedding_DataSubSet(Embedding_DataSubSet): 
    def __init__(self, dss: Canonical_DataSubSet, nperseg = 50, noverlap = 30):
        """Initialize the data subset converting from the canonical data subset"""
        self.embedding_name = "AccGyrSpectrogramEmbedding-{}-{}".format(nperseg,noverlap)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.samples = [self.convert_sample(i) for i in dss.samples]
        self.labels = dss.labels

    def display_sample(self, display_samples:list = [], label_to_str=None, sharey = True):
        nsamples = len(display_samples)
        fig, axs = plt.subplots(nrows=2, ncols=6*nsamples, figsize=(5*6*nsamples,5), sharey=sharey) # , sharey=True)
        for i in range(nsamples):
            sample_idx = display_samples[i]
            s = self.samples[sample_idx]
            l = self.labels[sample_idx]
            if label_to_str:
                sample_title = "Sample {} : Label {}".format(sample_idx, label_to_str(l))
            else:
                sample_title = "Sample {} : Label {}".format(sample_idx, l)

            def plot_charts(ax_0, ax_1, ax_2, ax_3, ax_4, ax_5,
                            ax_6, ax_7, ax_8, ax_9, ax_10, ax_11,
                            s, sample_title):
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

                ax_6.set_title(sample_title)
                ax_6.pcolormesh(s[2], s[1], s[0][3], shading='gouraud')
                ax_6.set_ylabel('Frequency [Hz]')
                ax_7.set_xlabel('Time [sec]')
                ax_7.set_ylabel('Frequency [Hz]')
                ax_7.imshow(s[0][3], cmap='hot', interpolation='nearest')

                ax_8.set_title(sample_title)
                ax_8.pcolormesh(s[2], s[1], s[0][4], shading='gouraud')
                ax_8.set_ylabel('Frequency [Hz]')
                ax_9.set_xlabel('Time [sec]')
                ax_9.set_ylabel('Frequency [Hz]')
                ax_9.imshow(s[0][4], cmap='hot', interpolation='nearest')

                ax_10.set_title(sample_title)
                ax_10.pcolormesh(s[2], s[1], s[0][5], shading='gouraud')
                ax_10.set_ylabel('Frequency [Hz]')
                ax_11.set_xlabel('Time [sec]')
                ax_11.set_ylabel('Frequency [Hz]')
                ax_11.imshow(s[0][5], cmap='hot', interpolation='nearest')

            if nsamples == 1: plot_charts(axs[0][0], axs[1][0], axs[0][1], axs[1][1], axs[0][2], axs[1][2],
                                          axs[0][3], axs[1][3], axs[0][4], axs[1][4], axs[0][5], axs[1][5],
                                          s, sample_title)
            else:             plot_charts(axs[0][0+6*i], axs[1][0+6*i], axs[0][1+6*i], axs[1][1+6*i], axs[0][2+6*i], axs[1][2+6*i],
                                          axs[0][3+6*i], axs[1][3+6*i], axs[0][4+6*i], axs[1][4+6*i], axs[0][5+6*i], axs[1][5+6*i],
                                          s, sample_title)

        plt.show()      
        
    def gen_spectrogram(self, s : Canonical_Sample):
        """TODO: Describe"""
        _, _, Sxx_acc_x = signal.spectrogram(s.acc.x.data, 100, nperseg=self.nperseg, noverlap=self.noverlap)
        _, _, Sxx_acc_y = signal.spectrogram(s.acc.y.data, 100, nperseg=self.nperseg, noverlap=self.noverlap)
        _, _, Sxx_acc_z = signal.spectrogram(s.acc.z.data, 100, nperseg=self.nperseg, noverlap=self.noverlap)
        _, _, Sxx_gyr_x = signal.spectrogram(s.gyr.x.data, 100, nperseg=self.nperseg, noverlap=self.noverlap)
        _, _, Sxx_gyr_y = signal.spectrogram(s.gyr.y.data, 100, nperseg=self.nperseg, noverlap=self.noverlap)
        f, t, Sxx_gyr_z = signal.spectrogram(s.gyr.z.data, 100, nperseg=self.nperseg, noverlap=self.noverlap)
        return ((Sxx_acc_x,Sxx_acc_y,Sxx_acc_z,Sxx_gyr_x,Sxx_gyr_y,Sxx_gyr_z), f, t)
    
    def convert_sample(self,s: Canonical_Sample) : return self.gen_spectrogram(s)

    def get_X(self): 
        return [np.hstack((s[0][0].ravel(), s[0][1].ravel(), s[0][2].ravel(), s[0][3].ravel(), s[0][4].ravel(), s[0][5].ravel())) for s in self.samples]
        #return [np.hstack((s[0][0].ravel(), s[0][1].ravel(), s[0][2].ravel(), 
        #        s[0][3].ravel(), s[0][4].ravel, s[0][5].ravel()))  for s in self.samples]

class AccGyrSpectrogramEmbedding_DataSet(Embedding_DataSet): 
    def __init__(self, nperseg = 50, noverlap = 30):
        """Initialize the dataset converting from the canonical dataset"""
        self.embedding_name = "AccGyrSpectrogramEmbedding-{}-{}".format(nperseg,noverlap)
        self.train      = None
        self.validation = None
        self.test       = None
        self.nperseg    = nperseg
        self.noverlap   = noverlap

    def import_from_canonical(self, ds: Canonical_DataSet):
        self.train      = AccGyrSpectrogramEmbedding_DataSubSet(ds.train, nperseg=self.nperseg, 
                                                                noverlap=self.noverlap)
        self.validation = AccGyrSpectrogramEmbedding_DataSubSet(ds.validation, nperseg=self.nperseg, 
                                                                noverlap=self.noverlap)
        self.test       = AccGyrSpectrogramEmbedding_DataSubSet(ds.test, nperseg=self.nperseg, 
                                                                noverlap=self.noverlap)
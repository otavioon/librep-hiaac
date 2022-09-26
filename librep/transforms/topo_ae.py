import numpy as np
import torch
from librep.estimators.ae.torch.models.topological_ae.topological_ae import TopologicallyRegularizedAutoencoder

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike
from torch.optim import Adam
import matplotlib.pyplot as plt

class TopologicalDimensionalityReduction(Transform):

    def __init__(self,
                 ae_model='ConvolutionalAutoencoder', ae_kwargs=None,
                 patience=10, num_epochs=1000, batch_size=128, input_shape = (-1, 1, 28, 28)):
        self.patience = patience
        self.num_epochs = num_epochs
        self.model = TopologicallyRegularizedAutoencoder(autoencoder_model=ae_model, ae_kwargs=ae_kwargs)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.max_loss = 10000
        self.loss_components_values = []

    def fit(self, X: ArrayLike, y: ArrayLike = None):
        self.loss_components_values = []
        data_loader = torch.utils.data.DataLoader(dataset=X, batch_size=self.batch_size, shuffle=True)
        patience = self.patience
        max_loss = self.max_loss
        loss_autoencoder_data_means = []
        loss_topo_error_data_means = []
        plot_last_ae_loss = []
        plot_last_topo_error = []
        plot_mean_ae_loss = []
        plot_mean_topo_error = []
        plot_full_ae_loss = []
        plot_full_topo_error = []
        for epoch in range(self.num_epochs):
            epoch_ae_loss = []
            epoch_topo_error = []
            self.model.train()
            # components_per_epoch = []
            # data_per_epoch = {'epoch':epoch, 'loss': [], 'loss_components': []}
            for data in data_loader:
                # print(data.shape)
                reshaped_data = np.reshape(data, self.input_shape)
                in_tensor = torch.Tensor(reshaped_data).float()
                loss, loss_components = self.model(in_tensor)
                # data_per_epoch['loss'].append(loss)
                # data_per_epoch['loss_components'].append(loss_components)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_ae_loss.append(loss_components['loss.autoencoder'].item())
                epoch_topo_error.append(loss_components['loss.topo_error'].item())
            # loss_autoencoder_data_means.append(np.mean(loss_autoencoder_data))
            # loss_topo_error_data_means.append(np.mean(loss_topo_error_data))
            
            # self.loss_components_values.append(data_per_epoch)
            last_loss = loss_components['loss.autoencoder']
            last_topo = loss_components['loss.topo_error']
            plot_last_ae_loss.append(loss_components['loss.autoencoder'].item())
            plot_last_topo_error.append(loss_components['loss.topo_error'].item())
            plot_mean_ae_loss.append(np.mean(epoch_ae_loss))
            plot_mean_topo_error.append(np.mean(epoch_topo_error))
            plot_full_ae_loss.append(epoch_ae_loss)
            plot_full_topo_error.append(epoch_topo_error)
            loss_per_epoch = np.mean(epoch_ae_loss) + np.mean(epoch_topo_error)
            ae_loss_per_epoch = np.mean(epoch_ae_loss)
            topo_loss_per_epoch = np.mean(epoch_topo_error)
            # loss_per_epoch = loss.item()
            # ae_loss_per_epoch = loss_components['loss.autoencoder'].item()
            # topo_loss_per_epoch = loss_components['loss.topo_error'].item()
            
            print(f'Epoch:{epoch+1}, Loss:{loss_per_epoch:.4f}, Loss-ae:{ae_loss_per_epoch:.4f}, Loss-topo:{topo_loss_per_epoch:.4f}')
            if max_loss < loss_per_epoch:
                if patience == 0:
                    break
                patience -= 1
            else:
                max_loss = loss_per_epoch
                patience = self.patience
        plt.title('Training considering last batch')
        plt.plot(plot_last_ae_loss, label='reconstruction error')
        plt.plot(plot_last_topo_error, label='topological error')
        plt.grid()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        plt.title('Training considering mean of batches')
        plt.plot(plot_mean_ae_loss, label='reconstruction error')
        plt.plot(plot_mean_topo_error, label='topological error')
        plt.grid()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        plt.title('Training considering all batches (boxplot)')
        plt.boxplot(plot_full_ae_loss)
        plt.boxplot(plot_full_topo_error)
        plt.grid()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        return self

    # TODO
    def transform(self, X: ArrayLike):
        self.model.eval()
        reshaped_data = np.reshape(X, self.input_shape)
        in_tensor = torch.Tensor(reshaped_data).float()
        # print('TRANSFORM', X.shape, X)
        return self.model.encode(in_tensor).detach().numpy()
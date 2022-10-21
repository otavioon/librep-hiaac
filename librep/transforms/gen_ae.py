import numpy as np
import torch
# from librep.estimators.ae.torch.models.topological_ae.topological_ae import TopologicallyRegularizedAutoencoder
from librep.estimators.ae.torch.models.generic_ae.generic_ae import GenericAutoencoder
from sklearn.model_selection import train_test_split
from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike
from torch.optim import Adam
import matplotlib.pyplot as plt

class GenericAEReduction(Transform):

    def __init__(self, patience=10, num_epochs=1000, batch_size=128, dimension=100):
        self.patience = patience
        self.num_epochs = num_epochs
        self.model = GenericAutoencoder(latent_space_dim=dimension)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.loss_function = torch.nn.MSELoss()
        self.batch_size = batch_size
        # self.input_shape = input_shape
        self.max_loss = 10000
    
    def fit_for_1000_epochs(self, X: ArrayLike, y: ArrayLike = None):
        return

    def fit(self, X: ArrayLike, y: ArrayLike = None):
        train_X, val_X, train_Y, val_Y = train_test_split(X, y, random_state=0, train_size = .8, stratify=y)
        train_data_loader = torch.utils.data.DataLoader(dataset=train_X, batch_size=self.batch_size, shuffle=True)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_X, batch_size=self.batch_size, shuffle=True)
        patience = self.patience
        max_loss = self.max_loss
        
        plot_train_loss = []
        plot_val_loss = []
        losses = []
        for epoch in range(self.num_epochs):
            epoch_train_loss = []
            epoch_val_loss = []
            self.model.train()
            for data in train_data_loader:
                
                in_tensor = torch.Tensor(data).float()
                reconstructed = self.model(in_tensor)
                loss = self.loss_function(reconstructed, data).float()
                self.optimizer.zero_grad()
                print(loss)
                loss.backward()
                self.optimizer.step()
                losses.append(loss)
                
                # reshaped_data = np.reshape(data, self.input_shape)
                # in_tensor = torch.Tensor(reshaped_data).float()
                # loss, loss_components = self.model(in_tensor)
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
                # epoch_train_loss.append(loss.item())
                # epoch_train_ae_loss.append(loss_components['loss.autoencoder'].item())
                # epoch_train_topo_error.append(loss_components['loss.topo_error'].item())
            # Verificar despues self.model()
            # for data in val_data_loader:
            #     reshaped_data = np.reshape(data, self.input_shape)
            #     in_tensor = torch.Tensor(reshaped_data).float()
            #     loss, loss_components = self.model(in_tensor)
            #     epoch_val_loss.append(loss.item())
            #     epoch_val_ae_loss.append(loss_components['loss.autoencoder'].item())
            #     epoch_val_topo_error.append(loss_components['loss.topo_error'].item())
            # plot_train_loss.append(np.mean(epoch_train_loss))
            # plot_train_ae.append(np.mean(epoch_train_ae_loss))
            # plot_train_topo.append(np.mean(epoch_train_topo_error))
            # plot_val_loss.append(np.mean(epoch_val_loss))
            # plot_val_ae.append(np.mean(epoch_val_ae_loss))
            # plot_val_topo.append(np.mean(epoch_val_topo_error))
            
            loss_per_epoch = loss.item()
            # ae_loss_per_epoch = np.mean(epoch_val_ae_loss)
            # topo_loss_per_epoch = np.mean(epoch_val_topo_error)
            
            print(f'Epoch:{epoch+1}, P:{patience}, Loss:{loss:.4f}')
            if max_loss < loss_per_epoch:
                if patience == 0:
                    break
                patience -= 1
            else:
                max_loss = loss_per_epoch
                patience = self.patience
        # plt.title('Training')
        # plt.plot(plot_train_ae, label='reconstruction error - train')
        # plt.plot(plot_train_topo, label='topological error - train')
        # plt.plot(plot_val_ae, label='reconstruction error - val')
        # plt.plot(plot_val_topo, label='topological error - val')
        # plt.grid()
        # plt.legend()
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.show()
        return self

    # TODO
    def transform(self, X: ArrayLike):
        self.model.eval()
        reshaped_data = np.reshape(X, self.input_shape)
        in_tensor = torch.Tensor(reshaped_data).float()
        # print('TRANSFORM', X.shape, X)
        return self.model.encode(in_tensor).detach().numpy()
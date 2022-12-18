import numpy as np
import torch
from librep.estimators.ae.torch.models.topological_ae.topological_ae import (
    TopologicallyRegularizedAutoencoder
)
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike
from torch.optim import Adam
import matplotlib.pyplot as plt
import pickle


class TopologicalDimensionalityReduction(Transform):

    def __init__(
        self, ae_model='ConvolutionalAutoencoder', ae_kwargs=None,
        lam=1., patience=None, num_epochs=500, batch_size=128,
        input_shape=(-1, 1, 28, 28), cuda_device_name='cuda:0',
        start_dim=180, latent_dim=10
    ):
        print('INIT_TopoDimRed')
        self.patience = patience
        self.num_epochs = num_epochs
        self.model_name = ae_model
        self.model_lambda = lam
        self.model_start_dim = start_dim
        self.model_latent_dim = latent_dim
        self.model = TopologicallyRegularizedAutoencoder(
            autoencoder_model=self.model_name,
            lam=self.model_lambda, ae_kwargs=ae_kwargs
        )
        # Setting cuda device
        self.cuda_device = torch.device(cuda_device_name)
        self.model.to(self.cuda_device)
        # Optimizer
        self.optimizer = Adam(self.model.parameters(),
                              lr=1e-3, weight_decay=1e-5)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.max_loss = 100000
        self.current = {
            'epoch': 0,
            'train_recon_error': None,
            'train_topo_error': None,
            'train_error': None,
            'val_recon_error': None,
            'val_topo_error': None,
            'val_error': None,
            'last_error': None
        }
        self.history = {
            'epoch': [],
            'train_recon_error': [],
            'train_topo_error': [],
            'train_error': [],
            'val_recon_error': [],
            'val_topo_error': [],
            'val_error': []
        }
        # self.loss_components_values = []

    def fit(self, X: ArrayLike, y: ArrayLike = None):
        # Splitting X into train and validation
        train_X, val_X, train_Y, val_Y = train_test_split(
            X, y, random_state=0,
            train_size=.8,
            stratify=y
        )
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_X,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_data_loader = torch.utils.data.DataLoader(
            dataset=val_X,
            batch_size=self.batch_size,
            shuffle=True
        )
        patience = self.patience
        max_loss = self.max_loss
        # Preparing for plot
        self.train_final_error = []
        self.train_recon_error = []
        self.train_topo_error = []

        self.val_final_error = []
        self.val_recon_error = []
        self.val_topo_error = []
        # Setting cuda
        # cuda0 = torch.device('cuda:0')
        for epoch in tqdm(range(self.num_epochs)):
            epoch_number = self.current['epoch'] + 1
            epoch_train_loss = []
            epoch_train_ae_loss = []
            epoch_train_topo_error = []
            epoch_val_loss = []
            epoch_val_ae_loss = []
            epoch_val_topo_error = []
            self.model.train()
            for data in train_data_loader:
                reshaped_data = np.reshape(data, self.input_shape)
                in_tensor = torch.tensor(reshaped_data, device=self.cuda_device).float()
                # self.model.to(cuda0)
                loss, loss_components = self.model(in_tensor)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss.append(loss.item())
                epoch_train_ae_loss.append(loss_components['loss.autoencoder'].item())
                epoch_train_topo_error.append(loss_components['loss.topo_error'].item())
            # Verificar despues self.model()
            for data in val_data_loader:
                reshaped_data = np.reshape(data, self.input_shape)
                in_tensor = torch.tensor(reshaped_data, device=self.cuda_device).float()
                loss, loss_components = self.model(in_tensor)
                epoch_val_loss.append(loss.item())
                epoch_val_ae_loss.append(loss_components['loss.autoencoder'].item())
                epoch_val_topo_error.append(loss_components['loss.topo_error'].item())
            self.current['epoch'] = self.current['epoch'] + 1
            self.current['train_recon_error'] = np.mean(epoch_train_ae_loss)
            self.current['train_topo_error'] = np.mean(epoch_train_topo_error)
            self.current['train_error'] = np.mean(epoch_train_loss)
            self.current['val_recon_error'] = np.mean(epoch_val_ae_loss)
            self.current['val_topo_error'] = np.mean(epoch_val_topo_error)
            self.current['val_error'] = np.mean(epoch_val_loss)
            self.history['epoch'].append(self.current['epoch'])
            self.history['train_recon_error'].append(self.current['train_recon_error'])
            self.history['train_topo_error'].append(self.current['train_topo_error'])
            self.history['train_error'].append(self.current['train_error'])
            self.history['val_recon_error'].append(self.current['val_recon_error'])
            self.history['val_topo_error'].append(self.current['val_topo_error'])
            self.history['val_error'].append(self.current['val_error'])
            loss_per_epoch = self.current['val_error']
            # loss_per_epoch = np.mean(epoch_val_ae_loss) + np.mean(epoch_val_topo_error)
            # ae_loss_per_epoch = np.mean(epoch_val_ae_loss)
            # topo_loss_per_epoch = np.mean(epoch_val_topo_error)
            # print(f'Epoch:{epoch+1}, P:{patience}, Loss:{loss_per_epoch:.4f}, Loss-ae:{ae_loss_per_epoch:.4f}, Loss-topo:{topo_loss_per_epoch:.4f}')
            if self.patience:
                if max_loss < loss_per_epoch:
                    if patience == 0:
                        break
                    patience -= 1
                else:
                    max_loss = loss_per_epoch
                    patience = self.patience
        
        return self
    
    def plot_training(self, title_plot=None):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_title('Training')
        if title_plot:
            ax.set_title(title_plot)
        ax.plot(self.history['train_recon_error'], label='reconstruction error - train', color='red')
        ax.plot(self.history['val_recon_error'], label='reconstruction error - val', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel("Reconstruction error", color="red", fontsize=14)
        ax.legend(loc=2)
        ax.set_ylim(bottom=0)

        ax2 = ax.twinx()
        ax2.plot(self.history['train_topo_error'], label='Topological error - train', color='blue')
        ax2.plot(self.history['val_topo_error'], label='Topological error - val', color='black')
        ax2.set_ylabel("Topological error", color="blue", fontsize=14)
        ax2.legend(loc=1)
        ax2.set_ylim(bottom=0)
        plt.grid()
        plt.show()
    
    def save(self, save_dir='data/', tag=None):
        model_name = self.model_name
        model_lambda = self.model_lambda
        model_start_dim = self.model_start_dim
        model_latent_dim = self.model_latent_dim
        model_epc = self.current['epoch']
        filename = '{}_{}_{}-{}_{}_{}.pkl'.format(
            model_name, model_lambda,
            model_start_dim, model_latent_dim,
            model_epc, tag)
        full_dir = save_dir + filename
        filehandler = open(full_dir, 'wb')
        pickle.dump(self, filehandler)
        filehandler.close()
        print('Saved ', full_dir)
        return full_dir
    
    def load(self, filename='data/test.pkl'):
        filehandler = open(filename, 'rb')
        self = pickle.load(filehandler)
        filehandler.close()
        print('Loaded ', filename)
    
    # TODO
    def transform(self, X: ArrayLike):
        # Setting cuda
        cuda0 = torch.device('cuda:0')
        self.model.eval()
        reshaped_data = np.reshape(X, self.input_shape)
        in_tensor = torch.tensor(reshaped_data, device=cuda0).float()
        return self.model.encode(in_tensor).cpu().detach().numpy()
    
    def inverse_transform(self, X: ArrayLike):
        # Setting cuda
        cuda0 = torch.device('cuda:0')
        self.model.eval()
        reshaped_data = np.reshape(X, (-1, 1, X.shape[-1]))
        in_tensor = torch.tensor(reshaped_data, device=cuda0).float()
        decoded = self.model.decode(in_tensor).cpu().detach().numpy()
        return np.reshape(decoded, (X.shape[0], -1))

    def transform_and_back(self, X: ArrayLike, plot_function):
        self.model.eval()
        reshaped_data = np.reshape(X, self.input_shape)
        in_tensor = torch.Tensor(reshaped_data).float()
        X_encoded = self.model.encode(in_tensor).detach().numpy()
        plot_function(X, X_encoded)
        return 
    
    def analize_patience(self, data):
        patiences = []
        patience = 0
        p = patience
        max_loss = np.max(data) + 1
        for index in range(1, len(data)):
            # print(index, data[index], p)
            if data[index] < max_loss:
                max_loss = data[index]
                p = patience
            else:
                if p == 0:
                    # print('PATIENCE', patience,' found in index', index, 'with value', data[index])
                    patiences.append(index)
                    patience +=1
                    p += 1
                p -= 1
        return (data, patiences)


class ConvolutionalTopologicalDimensionalityReduction(Transform):

    def __init__(
        self, ae_model='ConvolutionalAutoencoder', ae_kwargs=None,
        lam=1., patience=None, num_epochs=500, batch_size=128,
        input_shape=(-1, 1, 28, 28)
    ):
        self.patience = patience
        self.num_epochs = num_epochs
        self.model = TopologicallyRegularizedAutoencoder(
            autoencoder_model=ae_model,
            lam=lam, ae_kwargs=ae_kwargs
        )
        self.optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.max_loss = 10000
        # self.loss_components_values = []

    def fit(self, X: ArrayLike, y: ArrayLike = None, title_plot=None):
        # Splitting X into train and validation
        train_X, val_X, train_Y, val_Y = train_test_split(
            X, y, random_state=0,
            train_size = .8,
            stratify=y
        )
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_X,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_data_loader = torch.utils.data.DataLoader(
            dataset=val_X,
            batch_size=self.batch_size,
            shuffle=True
        )
        patience = self.patience
        max_loss = self.max_loss
        # Preparing for plot
        plot_train_loss = []
        plot_train_ae = []
        plot_train_topo = []

        plot_val_loss = []
        plot_val_ae = []
        plot_val_topo = []
        for epoch in range(self.num_epochs):
            epoch_train_loss = []
            epoch_train_ae_loss = []
            epoch_train_topo_error = []
            epoch_val_loss = []
            epoch_val_ae_loss = []
            epoch_val_topo_error = []
            self.model.train()
            for data in train_data_loader:
                reshaped_data = np.reshape(data, self.input_shape)
                in_tensor = torch.Tensor(reshaped_data).float()
                loss, loss_components = self.model(in_tensor)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss.append(loss.item())
                epoch_train_ae_loss.append(loss_components['loss.autoencoder'].item())
                epoch_train_topo_error.append(loss_components['loss.topo_error'].item())
            # Verificar despues self.model()
            for data in val_data_loader:
                reshaped_data = np.reshape(data, self.input_shape)
                in_tensor = torch.Tensor(reshaped_data).float()
                loss, loss_components = self.model(in_tensor)
                epoch_val_loss.append(loss.item())
                epoch_val_ae_loss.append(loss_components['loss.autoencoder'].item())
                epoch_val_topo_error.append(loss_components['loss.topo_error'].item())
            plot_train_loss.append(np.mean(epoch_train_loss))
            plot_train_ae.append(np.mean(epoch_train_ae_loss))
            plot_train_topo.append(np.mean(epoch_train_topo_error))
            plot_val_loss.append(np.mean(epoch_val_loss))
            plot_val_ae.append(np.mean(epoch_val_ae_loss))
            plot_val_topo.append(np.mean(epoch_val_topo_error))
            
            loss_per_epoch = np.mean(epoch_val_loss)
            # loss_per_epoch = np.mean(epoch_val_ae_loss) + np.mean(epoch_val_topo_error)
            ae_loss_per_epoch = np.mean(epoch_val_ae_loss)
            topo_loss_per_epoch = np.mean(epoch_val_topo_error)
            
            print(f'Epoch:{epoch+1}, P:{patience}, Loss:{loss_per_epoch:.4f}, Loss-ae:{ae_loss_per_epoch:.4f}, Loss-topo:{topo_loss_per_epoch:.4f}')
            if self.patience: 
                if max_loss < loss_per_epoch:
                    if patience == 0:
                        break
                    patience -= 1
                else:
                    max_loss = loss_per_epoch
                    patience = self.patience
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_title('Training')
        if title_plot:
            ax.set_title(title_plot)
        ax.plot(plot_train_ae, label='reconstruction error - train', color='red')
        ax.plot(plot_val_ae, label='reconstruction error - val', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel("Reconstruction error", color="red", fontsize=14)
        ax.legend(loc=2)
        ax.set_ylim(bottom=0)

        ax2 = ax.twinx()
        ax2.plot(plot_train_topo, label='Topological error - train', color='blue')
        ax2.plot(plot_val_topo, label='Topological error - val', color='black')
        ax2.set_ylabel("Topological error", color="blue", fontsize=14)
        ax2.legend(loc=1)
        ax2.set_ylim(bottom=0)

        plt.grid()
        # plt.legend()
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        plt.show()
        
        return self

    # TODO
    def transform(self, X: ArrayLike):
        self.model.eval()
        reshaped_data = np.reshape(X, self.input_shape)
        in_tensor = torch.Tensor(reshaped_data).float()
        return self.model.encode(in_tensor).detach().numpy()
    
    def inverse_transform(self, X: ArrayLike):
        self.model.eval()
        reshaped_data = np.reshape(X, (-1, 1, X.shape[-1]))
        in_tensor = torch.Tensor(reshaped_data).float()
        decoded = self.model.decode(in_tensor).detach().numpy()
        return np.reshape(decoded, (X.shape[0], -1))

    def transform_and_back(self, X: ArrayLike, plot_function):
        self.model.eval()
        reshaped_data = np.reshape(X, self.input_shape)
        in_tensor = torch.Tensor(reshaped_data).float()
        X_encoded = self.model.encode(in_tensor).detach().numpy()
        plot_function(X, X_encoded)
        return 
    
    def analize_patience(self, data):
        patiences = []
        patience = 0
        p = patience
        max_loss = np.max(data) + 1
        for index in range(1, len(data)):
            # print(index, data[index], p)
            if data[index] < max_loss:
                max_loss = data[index]
                p = patience
            else:
                if p == 0:
                    # print('PATIENCE', patience,' found in index', index, 'with value', data[index])
                    patiences.append(index)
                    patience +=1
                    p += 1
                p -= 1
        return (data, patiences)


class CustomTopoDimRedTransform(TopologicalDimensionalityReduction):
    def __init__(self,
                 model_name='DeepAE_custom_dim',
                 model_lambda=1,
                 patience=None,
                 num_epochs=175,
                 start_dim=180,
                 latent_dim=2,
                 batch_size=128,
                 cuda_device_name='cuda:0'):
        ae_kwargs = {
            'input_dims': (1, start_dim),
            'custom_dim': latent_dim
        }
        super().__init__(
            ae_model=model_name,
            ae_kwargs=ae_kwargs,
            lam=model_lambda,
            patience=patience,
            num_epochs=num_epochs,
            batch_size=batch_size,
            input_shape=(-1, 1, start_dim),
            cuda_device_name=cuda_device_name,
            start_dim=start_dim,
            latent_dim=latent_dim
        )
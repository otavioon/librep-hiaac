import numpy as np
import torch
from librep.estimators.ae.torch.models.topological_ae.topological_ae import TopologicallyRegularizedAutoencoder

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike
from torch.optim import Adam


class TopologicalDimensionalityReduction(Transform):

    def __init__(self,
                 ae_model='ConvolutionalAutoencoder', ae_kwargs=None,
                 patience=10, num_epochs=100, batch_size=128, input_shape = (-1, 1, 28, 28)):
        self.patience = patience
        self.num_epochs = num_epochs
        self.model = TopologicallyRegularizedAutoencoder(autoencoder_model=ae_model, ae_kwargs=ae_kwargs)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.max_loss = 10000

    def fit(self, X: ArrayLike, y: ArrayLike = None):
        data_loader = torch.utils.data.DataLoader(dataset=X, batch_size=self.batch_size, shuffle=True)
        patience = self.patience
        max_loss = self.max_loss
        for epoch in range(self.num_epochs):
            self.model.train()
            for data in data_loader:
                # print(data.shape)
                reshaped_data = np.reshape(data, self.input_shape)
                in_tensor = torch.Tensor(reshaped_data).float()
                loss, _ = self.model(in_tensor)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
            if max_loss < loss.item():
                if patience == 0:
                    break
                patience -= 1
            else:
                max_loss = loss.item()
        return self

    # TODO
    def transform(self, X: ArrayLike):
        self.model.eval()
        reshaped_data = np.reshape(X, self.input_shape)
        in_tensor = torch.Tensor(reshaped_data).float()
        # print('TRANSFORM', X.shape, X)
        return self.model.encode(in_tensor).detach().numpy()
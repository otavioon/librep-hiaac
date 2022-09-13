import numpy as np

from librep.estimators.ae.torch.models.topological_ae.topological_ae import TopologicallyRegularizedAutoencoder

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike
from torch.optim import Adam


class TopologicalDimensionalityReduction(Transform):

    def __init__(self,
                 ae_model='ConvolutionalAutoencoder',
                 patience=10, num_epochs=100):
        self.patience = patience
        self.num_epochs = num_epochs
        self.model = TopologicallyRegularizedAutoencoder(autoencoder_model=ae_model)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.max_loss = 10000

    def fit(self, X: ArrayLike, y: ArrayLike = None):
        patience = self.patience
        max_loss = self.max_loss
        for epoch in self.num_epochs:
            self.model.train()
            for i in range(len(X)):
                sample = X[i]
                loss, _ = self.model(sample)
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
        return self.model.encode(X).detach().numpy()
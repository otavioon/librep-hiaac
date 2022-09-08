from model_base import AutoencoderModel
from topological_signature_distance import TopologicalSignatureDistance
import model_submodules
import torch
import torch.nn as nn

class TopologicallyRegularizedAutoencoder(AutoencoderModel):
    """Topologically regularized autoencoder."""
    
    def __init__(self, lam=1., autoencoder_model='ConvolutionalAutoencoder',
                 ae_kwargs=None, toposig_kwargs=None):
        """Topologically Regularized Autoencoder.
        Args:
            lam: Regularization strength
            ae_kwargs: Kewords to pass to `ConvolutionalAutoencoder` class
            toposig_kwargs: Keywords to pass to `TopologicalSignature` class
        """
        super().__init__()
        self.lam = lam
        ae_kwargs = ae_kwargs if ae_kwargs else {}
        toposig_kwargs = toposig_kwargs if toposig_kwargs else {}
        self.topo_sig = TopologicalSignatureDistance(**toposig_kwargs)
        self.autoencoder = getattr(model_submodules, autoencoder_model)(**ae_kwargs)
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1),
                                              requires_grad=True)

    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances

    def forward(self, x):
        """Compute the loss of the Topologically regularized autoencoder.
        Args:
            x: Input data
        Returns:
            Tuple of final_loss, (...loss components...)
        """
        latent = self.autoencoder.encode(x)

        x_distances = self._compute_distance_matrix(x)

        dimensions = x.size()
        if len(dimensions) == 4:
            # If we have an image dataset, normalize using theoretical maximum
            batch_size, ch, b, w = dimensions
            # Compute the maximum distance we could get in the data space (this
            # is only valid for images wich are normalized between -1 and 1)
            max_distance = (2**2 * ch * b * w) ** 0.5
            x_distances = x_distances / max_distance
        else:
            # Else just take the max distance we got in the batch
            x_distances = x_distances / x_distances.max()

        latent_distances = self._compute_distance_matrix(latent)
        latent_distances = latent_distances / self.latent_norm

        # Use reconstruction loss of autoencoder
        ae_loss, ae_loss_comp = self.autoencoder(x)
#         print('TEST'*20)
#         print(self.topo_sig(x_distances, latent_distances))
        topo_error, topo_error_components = self.topo_sig(
            x_distances, latent_distances)

        # normalize topo_error according to batch_size
        batch_size = dimensions[0]
        topo_error = topo_error / float(batch_size) 
        loss = ae_loss + self.lam * topo_error
        loss_components = {
            'loss.autoencoder': ae_loss,
            'loss.topo_error': topo_error
        }
        loss_components.update(topo_error_components)
        loss_components.update(ae_loss_comp)
        return (
            loss,
            loss_components
        )

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)
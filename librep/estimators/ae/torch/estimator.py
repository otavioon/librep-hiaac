from librep.base.estimator import Estimator
from librep.config.type_definitions import ArrayLike

class TorchEstimator(Estimator):
    def __init__(model, loss, n_epochs, batch_size, metrics):
        pass

    def fit(self, X: ArrayLike, y: ArrayLike = None, **estimator_params) -> 'Estimator':
        """Fits the model function arround X. It takes some samples (X) and
        the respective labels (y) if the model is supervised.

        Parameters
        ----------
        X : ArrayLike
            An array-like representing the whole dataset with shape:
            (n_samples, n_features).
        y : ArrayLike
            The respective labels, with shape: (n_samples, ). This parameter is
            optional and may be used if needed.
        **estimator_params : type
            Optional data-dependent parameters.

        Returns
        -------
        'Estimator'
            The estimator (self).

        """

        raise NotImplementedError

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict or project the values of the samples (X).

        Parameters
        ----------
        X : ArrayLike
            An array-like of samples, with shape: (n_features, ).

        Returns
        -------
        ArrayLike
            An array-like describing the respective labels predicted for each
            samples.

        """

        raise NotImplementedError

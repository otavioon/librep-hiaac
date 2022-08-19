from librep.config.type_definitions import ArrayLike
from librep.base.parametrizable import Parametrizable


# Wrap around scikit learn base API
# Borrowed from Sklearn API

class Estimator(Parametrizable):
    """An object which manages the estimation and decoding of a model.
    """

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

    def fit_predict(self,
                    X: ArrayLike,
                    y: ArrayLike = None,
                    **estimator_params) -> ArrayLike:
        """Chain fit and predict methods, togheter. It fits the model and
        predict the samples of X.

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
        ArrayLike
            An array-like with shape (n_samples, ) describing the respective
            labels predicted for each samples.

        """
        self.fit(X, y, **estimator_params)
        return self.predict(X)

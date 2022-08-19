from librep.config.type_definitions import ArrayLike
from librep.base.parametrizable import Parametrizable


# Wrap around scikit learn base API
# Borrowed from Sklearn API

class Transform(Parametrizable):
    """For filtering or modifying the data, in a supervised or unsupervised way.
    `fit` allows implementing parametrizable transforms. This method sees the
    whole dataset. `transform` allows transforming each sample.
    """

    def fit(self, X: ArrayLike, y: ArrayLike = None, **fit_params) -> 'Transform':
        """Fit the transformation with information of the whole dataset.

        Parameters
        ----------
        X : ArrayLike
            An array-like representing the whole dataset with shape:
            (n_samples, n_features).
        y : ArrayLike
            The respective labels, with shape: (n_samples, ). This parameter is
            optional and may be used if needed.
        **fit_params : type
            Optional data-dependent parameters.

        Returns
        -------
        'Transform'
            The transform (self).

        """
        raise NotImplementedError

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transforms a single sample.

        Parameters
        ----------
        X : ArrayLike
            An array-like of sample with shape (n_features, ).

        Returns
        -------
        ArrayLike
            An array-like with the transformed sample.

        """
        raise NotImplementedError

    def fit_transform(self, X: ArrayLike, y: ArrayLike = None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class InvertibleTransform(Transform):
    """Denotes a invertible transform."""

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        """Perform the inverse transform on data.

        Parameters
        ----------
        X : ArrayLike
            An array-like of sample with shape (n_features, ).

        Returns
        -------
        ArrayLike
            An array-like with the transformed sample.

        """
        raise NotImplementedError

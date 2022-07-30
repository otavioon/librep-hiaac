from librep.config.type_definitions import ArrayLike
from librep.base.parametrizable import Parametrizable


# Wrap around scikit learn base API
class Transform(Parametrizable):

    def fit(self, X: ArrayLike, y: ArrayLike = None, **fit_params) -> None:
        raise NotImplementedError

    def transform(self, X: ArrayLike):
        raise NotImplementedError

    def fit_transform(self, X: ArrayLike, y: ArrayLike = None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class InvertibleTransform(Transform):

    def inverse_transform(self, X: ArrayLike):
        raise NotImplementedError


class PartialTransform(Transform):

    def partial_fit(self, X: ArrayLike, y: ArrayLike = None, **fit_params):
        raise NotImplementedError


class DifferentiableTransform(Transform):

    def forward(self, *args, **kwargs):
        pass
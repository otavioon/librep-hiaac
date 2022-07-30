from librep.config.type_definitions import ArrayLike
from librep.base.parametrizable import Parametrizable


# Wrap around scikit learn base API
class Estimator(Parametrizable):

    def fit(self, X: ArrayLike, y: ArrayLike = None, **estimator_params):
        raise NotImplementedError

    def predict(self, X: ArrayLike):
        raise NotImplementedError

    def fit_predict(self,
                    X: ArrayLike,
                    y: ArrayLike = None,
                    **estimator_params):
        raise NotImplementedError
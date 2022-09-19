import numpy as np

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike


class RemoveFrequencies(Transform):
    """Performs the remotion of frequencies at one sample.

    Parameters
    ----------
    acc : bool
        Return the accelerometer transformed sample (the default is True).
    gyr : bool
        Return the gyroscope transformed sample (the default is True).
    interval : []
        It's a list with the columns to be remove (the default is a empty list).

    Examples
    ----------

    """

    def __init__(self,
                 acc: bool = True,
                 gyr: bool = True,
                 interval: list = []):

        self.acc = acc
        self.gyr = gyr
        self.interval = interval

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transform a sample removing frequencies.

        Parameters
        ----------
        X : ArrayLike
            The samples to be transfomed with shape (n_features, n_features, ).

        Returns
        -------
        ArrayLike
            The transformed sample.

        """
        if self.interval == []:
            return X
        datas = []
        for sample in X:
            data = np.delete(sample, self.interval, axis=0)
            datas.append(data)

        return np.array(datas)
